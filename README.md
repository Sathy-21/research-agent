# Research Agent

An autonomous research agent that answers a single research question by planning sub-questions, researching each one on the web, and composing a grounded report with sources.

## What it does

Given one research question, the agent:

1. **Plans** — uses an LLM to break the question into 3–6 focused sub-questions.
2. **Researches each sub-question** — runs a web search, fetches and extracts the readable text of the top results, and writes a short, grounded answer that tracks which source URLs it used.
3. **Composes** — merges the sub-answers into a single final report and ends with a numbered list of all unique sources used.

It then prints the original question, the generated sub-questions, the final report, and the source list.

## Stack

- **Python** — organized as a small package (`research_agent/`) with separate modules per concern.
- **Groq API** (free tier) — every step runs on `llama-3.3-70b-versatile`, so the project costs nothing to run and has a daily request allowance large enough to iterate on a multi-call agent. See [DECISIONS.md](DECISIONS.md) for the provider history.
- **Tavily** — web search.
- **trafilatura** — readable-text extraction from fetched pages.
- **python-dotenv** — loads API keys from a local `.env` file.

## Setup

```bash
# 1. (Recommended) create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API keys
copy .env.example .env       # Windows  (use `cp` on macOS / Linux)
# then edit .env and fill in your real keys
```

You need two keys in `.env`:

- `GROQ_API_KEY` — from <https://console.groq.com/keys> (free).
- `TAVILY_API_KEY` — from <https://app.tavily.com/> (free tier available).

The `.env` file is gitignored and never committed.

## Run

```bash
python main.py "What are the main approaches to carbon capture, and how do they compare on cost?"
```

Or run with no argument to be prompted for the question interactively:

```bash
python main.py
```

The report prints to **stdout**; diagnostic logging goes to **stderr**. Turn logging up
with `--verbose` (DEBUG) or set `LOG_LEVEL` (e.g. `LOG_LEVEL=WARNING`):

```bash
python main.py --verbose "..."
LOG_LEVEL=WARNING python main.py "..."
```

## Architecture

The flow is **plan → per-sub-question pipeline → compose → verify**, orchestrated by `research_agent/agent.py`. Read `run_research` top to bottom to follow the entire control flow.

```
question
   │
   ▼
[ validate ]  ── trim/normalize, reject empty or too-long (before any API call)  (Phase 4)
   │
   ▼
[ planner.make_plan ]  ── LLM ──▶  3–6 sub-questions          (Phase 2)
   │                       └─ fatal if no usable plan (PlanningError)
   │
   ▼  for each sub-question (skipped if budget/time exhausted):  (per-sub-question pipeline)
   ├─ retrieval.gather_sources    ── Tavily search ──▶ top 3 URLs
   │                              ── trafilatura ────▶ extracted text (bad pages skipped)
   ├─ relevance.filter_sources    ── LLM (1 batched) ──▶ keep only on-topic sources  (Phase 3A)
   │                                 └─ if none relevant, skip this sub-question
   └─ synthesis.answer_subquestion ── LLM ──▶ grounded answer + the sources it used
   │      (a sub-question that fails after retries is SKIPPED; the run continues)     (Phase 4)
   │
   ▼  (fatal only if ZERO sub-questions succeeded)
[ compose.compose_report ]  ── LLM ──▶  narrative report body
   │
   ▼
[ verify.verify_report ]  ── LLM (1 batched) ──▶ per-claim supported/unsupported   (Phase 3B)
   │   ├─ flag_unsupported: append a clearly marked "Unverified claims" section
   │   └─ grounding_summary: total / supported / percent grounded
   ▼
report body + coverage note (dropped sub-questions) + deduped numbered source list
   │
   ▼
ResearchResult  ──▶  printed by main.py;  run summary logged

All LLM and web-search calls go through retries.py (retry transient 503/429/timeout,
fail fast on 400/401), so a transient blip never discards completed work.            (Phase 4)
```

Source **text** (not just URLs) is threaded from `retrieval` through `synthesis` (each `AnsweredSubquestion` keeps the `Source` objects it used) into `verify`, so claims are checked against the same text the report was built from.

### Modules

| Module | Responsibility |
| --- | --- |
| `config.py` | Loads/validates API keys from `.env`; defines model choices and run limits. |
| `llm.py` | Thin helpers around the Groq API (plain-text and JSON replies). |
| `planner.py` | Phase 2 — decomposes the question into sub-questions. |
| `retrieval.py` | Web search (Tavily) + readable-text extraction (trafilatura). |
| `relevance.py` | Phase 3A — one batched LLM call per sub-question to drop off-topic sources before synthesis. |
| `synthesis.py` | Writes a grounded answer to one sub-question and tracks its sources. |
| `compose.py` | Writes the narrative report body; provides the deduped source-list helpers. |
| `verify.py` | Phase 3B — checks the report's claims against the source text, flags unsupported ones, and reports a grounding metric. |
| `retries.py` | Phase 4 — shared transient-failure retry layer (exponential backoff + jitter, honours server retry hints) used by every LLM and web-search call. |
| `agent.py` | Orchestrates the whole flow; enforces input/cost/time guardrails and partial-failure resilience; logs progress and the run summary. |
| `main.py` | CLI entry point; configures logging (`--verbose` / `LOG_LEVEL`), prints the report and grounding summary. |

### Robustness & observability (Phase 4)

- **Transient-failure retries** — every LLM and web-search call routes through `retries.py`, which retries only transient errors (HTTP 429/408/5xx, timeouts, connection errors) with capped exponential backoff + jitter, honouring a server `Retry-After` hint when present. Permanent errors (400/401/403/404/422) fail fast. The retry policy lives in one place, so all callers benefit.
- **Partial-failure resilience** — if a single sub-question fails (retries exhausted) or finds nothing relevant, it is skipped and the run continues. The report is composed from whatever succeeded, with a "Coverage note" listing the dropped sub-questions and why. The run is fatal only if planning fails outright or zero sub-questions succeed.
- **Input guardrails** — the question is trimmed/normalized and rejected if empty or absurdly long (`MAX_QUESTION_CHARS`) *before* any API call, with a clear message and non-zero exit code.
- **Cost & time guardrails** — a `Budget` caps total searches (`MAX_SEARCHES`) and LLM calls (`MAX_LLM_CALLS`); a wall-clock deadline (`MAX_RUN_SECONDS`) caps total run time. Hitting either is logged and handled gracefully (compose what's done), never a crash.
- **Observability** — diagnostics use the `logging` module (to stderr): phases entered, per-sub-question progress, retries/backoffs, sources kept vs filtered, claims verified, and budget usage. A concise **run summary** is logged at the end: total LLM calls, total searches, wall-clock time, sub-questions succeeded vs skipped, and grounding percentage.
- Existing graceful-failure behaviour is preserved: dead pages, unparseable JSON, and all-irrelevant sub-questions are all skipped; a missing API key fails fast with a clear message.
