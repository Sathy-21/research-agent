# Decisions

A running log of notable technical decisions.

## Moved the LLM provider from Anthropic to Google Gemini (free tier)

**What:** Switched all LLM calls from the Anthropic API (Claude) to the Google Gemini
API via Google AI Studio, using `gemini-2.5-flash` for every step.

**Why:** Gemini's free tier (Google AI Studio) lets the project run at zero cost, which
is the goal for a portfolio project that should be cheap for anyone to try. The switch
was cheap to make because all LLM access is funneled through `research_agent/llm.py` —
only that file and `config.py` changed; the planner, synthesis, compose, and agent loop
kept their interfaces.

**Why Flash everywhere (no model split):** The earlier Anthropic version used a cheaper
model for the high-volume steps and a stronger one (Sonnet) for the single compose call.
The equivalent stronger Gemini model would be Gemini 2.5 Pro, but on the free tier Pro
has very low rate limits that a multi-call research run can exhaust, and — unlike Flash —
Pro cannot disable "thinking", which makes the output-token budget less predictable.
Flash for every step keeps runs free, fast, and within rate limits. The three model
constants in `config.py` are still separate, so a stronger compose model can be dropped
in later without touching other code.

**Thinking disabled:** Gemini 2.5 Flash is a thinking model by default, and thinking
tokens are drawn from `max_output_tokens`. With a small budget (e.g. the planner's), the
model could spend the entire budget thinking and return no text. `llm.py` disables
thinking (`thinking_budget=0`, supported on Flash) so the token budget goes entirely to
the answer.

**Unchanged:** The graceful-failure behaviour (skip dead sources / empty searches /
malformed JSON, fail fast on a missing key) and the per-run cost cap (`Budget` in
`agent.py`) are provider-independent and still apply.
