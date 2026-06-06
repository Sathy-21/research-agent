# Decisions

A running log of notable technical decisions.

## Moved the LLM provider from Anthropic to Google Gemini (free tier)

**What:** Switched all LLM calls from the Anthropic API (Claude) to the Google Gemini
API via Google AI Studio, using `gemini-2.5-flash-lite` for every step.

**Why:** Gemini's free tier (Google AI Studio) lets the project run at zero cost, which
is the goal for a portfolio project that should be cheap for anyone to try. The switch
was cheap to make because all LLM access is funneled through `research_agent/llm.py` —
only that file and `config.py` changed; the planner, synthesis, compose, and agent loop
kept their interfaces.

**Why Flash-Lite everywhere (no model split):** The earlier Anthropic version used a
cheaper model for the high-volume steps and a stronger one (Sonnet) for the single
compose call. The equivalent stronger Gemini model would be Gemini 2.5 Pro, but on the
free tier Pro has very low rate limits that a multi-call research run can exhaust, and —
unlike Flash-Lite — Pro cannot disable "thinking", which makes the output-token budget
less predictable. Flash-Lite for every step keeps runs free, fast, and within rate
limits. The three model constants in `config.py` are still separate, so a stronger
compose model can be dropped in later without touching other code.

**Thinking disabled:** Gemini 2.5 Flash-Lite is a thinking model by default, and thinking
tokens are drawn from `max_output_tokens`. With a small budget (e.g. the planner's), the
model could spend the entire budget thinking and return no text. `llm.py` disables
thinking (`thinking_budget=0`, supported on Flash-Lite) so the token budget goes entirely
to the answer.

**Unchanged:** The graceful-failure behaviour (skip dead sources / empty searches /
malformed JSON, fail fast on a missing key) and the per-run cost cap (`Budget` in
`agent.py`) are provider-independent and still apply.

## Phase 3: grounding via relevance filtering + claim verification

Added two steps to stop the agent trusting and repeating bad sources. Each addresses a
specific failure seen in a real test run.

**Relevance filtering — `relevance.py` (before synthesis).**
*Failure it defends against:* a sub-question's search pulled in off-topic sources
(vision/multimodal papers for a question about text-retrieval systems), and the
synthesizer summarized them anyway because nothing checked relevance. Now, for each
sub-question, one batched LLM call judges which gathered sources are actually relevant
and drops the rest *before* synthesis, so the answer is built only from on-topic text.
It is one call per sub-question (snippets of all sources in a single prompt), not one
call per source, to stay call-efficient on the rate-limited free tier. If filtering
removes every source, the sub-question is skipped rather than synthesized from nothing.

**Claim verification — `verify.py` (after compose).**
*Failure it defends against:* a low-quality, off-topic source (a generic library
fact-checking guide) was retrieved, trusted, and woven into a technical report. Now,
after the report is composed, one batched LLM call extracts the report's factual claims
and checks each against the actual retrieved source text, marking it supported or
unsupported. This required threading the source **text** (not just URLs) through to the
verifier: each `AnsweredSubquestion` carries the `Source` objects it used, and
`agent.py` passes the deduped set into `verify_report`.

*Flag vs. delete:* unsupported claims are **flagged** in an appended "Unverified claims"
section rather than deleted. Surgically removing sentences from finished prose would
need a second LLM rewrite call (cost on a rate-limited tier) and risks mangling the
text; flagging keeps the report intact and is transparent about what couldn't be
grounded. A one-line **grounding summary** (total claims / supported / percent grounded)
is printed after the report as a metric to build on in Phase 5.

**Budget impact:** new calls are counted in `Budget`. Worst case (6 sub-questions) is
now 1 plan + 6×(1 relevance + 1 synthesis) + 1 compose + 1 verification = 15 LLM calls,
so `MAX_LLM_CALLS` was raised from 12 to 16. Verification is kept to a single batched
call (all claims at once), not one call per claim.

## Moved the LLM provider from Google Gemini to Groq (free tier)

**What:** Switched all LLM calls from the Gemini API to the Groq API, using
`llama-3.3-70b-versatile` for every step. The JSON steps use Groq's JSON mode
(`response_format={"type": "json_object"}`).

**Why:** Gemini's free tier daily request quota was too small to iterate on this agent —
only ~20 requests/day for Flash-Lite on this account, and a single research run makes up
to ~15 LLM calls, so barely one run per day was possible. Groq's free tier offers a much
larger daily allowance, making iteration practical while still costing nothing.

**Why it was a one-module change:** all model access goes through `llm.py`'s two helpers
(`complete_text`, `complete_json`) with provider-independent signatures. Swapping
providers meant rewriting only `llm.py` (the request/JSON-mode mechanics) and `config.py`
(key name + model constants); the planner, relevance filter, synthesis, compose,
verifier, and agent loop were untouched apart from the client's type annotation. This is
the third provider on the same interface (Anthropic → Gemini → Groq), which is the
payoff of keeping the provider behind a thin seam.

**Unchanged:** the graceful-failure handling, the relevance filter, the verifier, and
the `Budget` cap all carry over without change — they are provider-independent.

## JSON output *shape* was an implicit coupling that broke under Groq

The provider swap was clean at the function-signature level (`llm.py`'s two helpers were
untouched by callers), but it surfaced a subtler coupling: the **shape** of the JSON the
model returns. Gemini happily returned a bare array for the planner, so the planner
iterated the parsed value directly. Groq's JSON mode only returns a top-level **object**,
so the reply came back as `{"sub_questions": [...]}` — and iterating that object yielded
its *keys*, producing a single bogus sub-question equal to the literal string
`"sub_questions"`.

The lesson: keeping the provider behind a thin interface isolates the *call mechanics*
but not the *data contract*. The fix makes parsing defensive about shape rather than
assuming one provider's convention:

- Added `llm.extract_list`, which returns the list whether the reply is a bare array or
  an object wrapping the array under a key (the first list-valued field). All
  list-expecting callers (planner, relevance, verify) use it; synthesis, which genuinely
  expects an object, now explicitly requires a dict.
- The planner validates the result (non-empty, at least 2 sub-questions, not a lone
  wrapper-key token), retries once on an invalid/garbage plan, and otherwise raises a
  clear `PlanningError` instead of proceeding — so a bad plan can never silently poison
  the rest of the run.

## Phase 4: robustness, guardrails, and observability

**Retry strategy (`retries.py`).** A live run crashed on a transient API blip, discarding
work already done. All LLM and web-search calls now route through one shared
`call_with_retries` helper (no copy-paste), giving application-level resilience on top of
the SDKs' own handling.

- *Retryable (transient):* HTTP 429 (rate limit), 408/425, and 5xx (500/502/503/504),
  plus timeouts and connection errors. These are server-side or load-related and usually
  succeed on a retry — exactly the class that shouldn't throw away completed work.
- *Not retried (permanent):* 400 (bad request), 401/403 (auth/bad key), 404, 409, 422.
  Retrying these only burns time and quota because the request itself is wrong; they fail
  fast with a clear logged message.
- *Backoff:* capped exponential (1s, 2s, 4s … up to `RETRY_MAX_DELAY`) with up to 25%
  jitter, limited to `RETRY_MAX_ATTEMPTS` total attempts. If the error carries a server
  hint (`Retry-After` header or a `retry_after` attribute — Groq and Gemini both provide
  one), that delay is honoured instead of the computed backoff.
- *Classification is by duck-typing* (status code + exception class-name hints) rather
  than importing each SDK's exception types, so the layer stays provider-agnostic — the
  same reason the provider swaps were cheap.

**Partial-failure policy.** A single bad sub-question must not sink the whole run. Each
sub-question's search → relevance → synthesis runs in a try/except; if it fails after
retries (or finds no relevant sources), it is recorded in a `skipped` list with a reason
and the loop continues. The report is composed from whatever succeeded and ends with a
"Coverage note" naming the dropped sub-questions, so the output is honest about gaps.
The run is fatal only when planning fails outright (`PlanningError`) or zero sub-questions
succeed (`NoResultsError`) — there is genuinely nothing to report in those cases.

**Time guardrail.** A wall-clock deadline (`MAX_RUN_SECONDS`) is checked cooperatively
between sub-questions rather than via signals/threads — this is portable (notably on
Windows, which lacks `SIGALRM`) and can't leave orphaned threads. On timeout the loop
stops, the remaining sub-questions are marked skipped, and whatever finished is still
composed and flagged as cut short. (Bounded retry sleeps and the SDK's own per-call
timeouts keep any single call from hanging indefinitely.)

**Logging approach.** Diagnostics use the standard `logging` module, emitted to **stderr**
so the user-facing report on **stdout** stays clean and machine-pipeable. Level is
controlled by `--verbose` (DEBUG) or the `LOG_LEVEL` env var (default INFO). Logged:
each phase entered, per-sub-question progress, retries/backoffs taken, sources kept vs
filtered, claims verified, budget usage, and a one-line **run summary** at the end (LLM
calls, searches, wall-clock time, sub-questions succeeded vs skipped, grounding %).

## Phase 5: trustworthy verifier extraction + evaluation harness

**The problem.** The grounding percentage was inconsistent and sometimes meaningless.
The root cause was the claim-**extraction** step, not the support check: the verifier
sometimes invented strawman claims (assertions the report never made) and marked them
unsupported, dragging the percentage down for no real reason. A grounding number is only
trustworthy if the claims being checked are claims the report actually makes.

**The fix (`verify.py`).**
- *Strict extraction prompt.* Extraction is now an explicit, emphatic first step:
  extract ONLY the report's own assertions, faithfully paraphrased, each a single
  checkable statement, at a consistent granularity (named techniques/numbers/entities,
  not just summary sentences) — never invent, generalize, negate, or combine. The
  support check against retrieved source text is unchanged.
- *Fabrication instrument.* Because a prompt alone can't be trusted, each extracted claim
  is also tested for whether it genuinely appears in the report, via a deterministic
  word-overlap heuristic (`claim_in_report`) — no extra LLM call. `VerificationReport`
  exposes `fabricated` (claims not in the report). A faithful paraphrase shares most of
  its content words with the report; a strawman shares few. It is a proxy, not a perfect
  entailment check, but it makes fabrication countable instead of silently skewing the
  metric.

**What the eval measures (and doesn't).** `eval/run_eval.py` runs the full agent over a
fixed benchmark of AI/RAG/hallucination questions and records, per question: grounding %,
claim counts (total/supported/fabricated), sub-questions succeeded vs skipped, LLM/search
calls, and elapsed time, with aggregate means. It deliberately has **no gold answers**:
it measures **grounding and process quality** (are the report's claims supported by the
sources, how did the pipeline behave) — NOT the factual correctness of the answers, which
would need curated ground truth. The benchmark stays in the tested domain so searches
return real sources.

**Before/after comparison.** Rather than delete the old prompt, it is kept behind
`VERIFIER_MODE=old|new` (read from the environment by `config.verifier_mode()`), so the
eval can run once in each mode and `eval/compare.py` prints mean grounding before vs
after plus the fabricated-claim counts each mode produced. This turns "the metric feels
untrustworthy" into a measured before/after — the strict prompt should show higher, more
stable grounding and far fewer fabricated claims.

**Free-tier discipline.** The runner is sequential, reuses the existing retry/backoff,
and pauses `--delay` seconds between questions so a full run doesn't trip Groq's
per-minute cap; `--limit N` runs a small subset first.

## Phase 5 hotfix: per-call token-budget cap (HTTP 413)

**The bug.** During eval, the verifier crashed with HTTP 413 "request too large": on one
question it packed all retrieved source text plus the report into a single prompt and
requested ~14,447 tokens. Groq's free tier has a **per-minute token limit (12,000 TPM)**,
and — crucially — a *single request* that exceeds it is rejected outright with 413, no
matter how long you wait. This is a per-request size problem, not a request-count or
rate-over-time problem (those surface as 429, which the retry layer already backs off on).
The retry layer correctly treated 413 as permanent (no point retrying an unchanged
over-size request); the real fix is to never send an over-size request.

**The cap.** `config.MAX_SOURCE_CONTEXT_CHARS` (default 12,000 chars ≈ 3,000 tokens)
bounds the largest variable part of a prompt — the concatenated source text. A shared
helper `retrieval.render_source_context(sources, budget)` distributes the budget evenly
across sources and truncates each source's text (rather than dropping whole sources, so
every source stays represented), returning whether truncation occurred. Applied to:
- **verify** — the offender, which aggregates the text of *every* unique source in the
  report; and
- **synthesis** — which concatenates a sub-question's sources.
- **compose** does *not* concatenate raw source text (it uses the already-short
  sub-answers), so it isn't the usual offender, but its findings block is capped to the
  same budget as a last-resort guard. Relevance already used tiny per-source snippets.

Why chars, not a real token count: we don't ship a Llama tokenizer, and ~4 chars/token is
a good, dependency-free approximation. The default leaves generous headroom below 12,000
for the report, prompt scaffolding, and the reply (and verify's completion was lowered to
2,048 tokens). **To tune:** lower `MAX_SOURCE_CONTEXT_CHARS` if 413s persist, raise it for
richer context.

**Graceful fallback.** If a verify request is *still* rejected as 413 after trimming, the
verifier halves the budget and retries once; if it still fails, it returns an empty result
marked `partial=True` instead of crashing. Any run whose source text had to be truncated
is also marked `partial`, and the grounding summary says so — the number is honest about
having seen only a subset. 413 is now also listed explicitly as permanent in `retries.py`.

## Observation: grounding % partly reflects source quality, not just the verifier

In the 5-question eval (new verifier mode), per-question grounding varied widely
(33%–91%). This largely tracked **source availability**, not a verifier flaw. Questions
that pulled paywalled or otherwise inaccessible sources (publisher 403s, redirect loops)
yielded less usable extracted text, which triggered the partial-verification fallback —
grounding was then checked against a truncated subset of source text, lowering the
grounding %.

Takeaway: grounding % is partly a function of retrieval/source quality, not only verifier
behaviour. It is a known limitation of measuring grounding on free-tier web retrieval: a
low score can mean "the report is poorly supported" *or* "the sources we could actually
fetch were thin". The per-run `partial` flag surfaces when the latter is in play.
