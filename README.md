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

## Architecture

The flow is **plan → per-sub-question pipeline → compose → verify**, orchestrated by `research_agent/agent.py`. Read `run_research` top to bottom to follow the entire control flow.

```
question
   │
   ▼
[ planner.make_plan ]  ── LLM ──▶  3–6 sub-questions          (Phase 2)
   │
   ▼  for each sub-question:                                  (per-sub-question pipeline)
   ├─ retrieval.gather_sources    ── Tavily search ──▶ top 3 URLs
   │                              ── trafilatura ────▶ extracted text (bad pages skipped)
   ├─ relevance.filter_sources    ── LLM (1 batched) ──▶ keep only on-topic sources  (Phase 3A)
   │                                 └─ if none relevant, skip this sub-question
   └─ synthesis.answer_subquestion ── LLM ──▶ grounded answer + the sources it used
   │
   ▼
[ compose.compose_report ]  ── LLM ──▶  narrative report body
   │
   ▼
[ verify.verify_report ]  ── LLM (1 batched) ──▶ per-claim supported/unsupported   (Phase 3B)
   │   ├─ flag_unsupported: append a clearly marked "Unverified claims" section
   │   └─ grounding_summary: total / supported / percent grounded
   ▼
report body + deduped numbered source list (built in code)
   │
   ▼
ResearchResult  ──▶  printed by main.py (question, sub-questions, report, grounding summary)
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
| `agent.py` | Orchestrates the whole flow and enforces the cost budget. |
| `main.py` | CLI entry point; prints the question, sub-questions, report, and grounding summary. |

### Cost & robustness guardrails

- A `Budget` (in `agent.py`) caps total searches (`MAX_SEARCHES`) and LLM calls (`MAX_LLM_CALLS`) per run, so a single run can't spiral in cost. Limits live in `config.py`.
- Failures degrade gracefully rather than crashing: a search that returns nothing, a page that won't load, or malformed model output is skipped, and the agent continues with whatever it has. A missing API key fails fast with a clear message.
