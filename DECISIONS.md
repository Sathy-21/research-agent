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
