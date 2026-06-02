"""Configuration: API-key loading, model choices, and run limits.

All secrets are read from environment variables (loaded from a local .env file via
python-dotenv) — nothing is hardcoded. Call `load_settings()` to get validated keys.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# --- Model choices ---------------------------------------------------------------
# We use Gemini's free tier (Google AI Studio) so the project costs nothing to run.
# We use the Flash model for every step rather than splitting in a stronger model for
# compose. The sensible stronger free-tier model would be Gemini 2.5 Pro, but on the
# free tier Pro has very low rate limits that a multi-call research run can exhaust,
# and (unlike Flash) it cannot disable "thinking", which makes token budgeting less
# predictable. Flash everywhere keeps runs free, fast, and within rate limits. The
# three constants are kept separate so a stronger compose model can be dropped in
# later without touching the rest of the code.
PLANNER_MODEL = "gemini-2.5-flash"
SYNTHESIS_MODEL = "gemini-2.5-flash"
COMPOSE_MODEL = "gemini-2.5-flash"

# --- Pipeline knobs --------------------------------------------------------------
MIN_SUBQUESTIONS = 3
MAX_SUBQUESTIONS = 6
RESULTS_PER_SUBQUESTION = 3   # fetch + extract the top N web-search hits per sub-question
MAX_SOURCE_CHARS = 6000       # truncate each extracted page to bound token cost

# --- Cost guardrails -------------------------------------------------------------
# Hard ceilings so a single run can never spiral in cost, regardless of how many
# sub-questions the planner returns or how the loop behaves.
MAX_SEARCHES = 8
MAX_LLM_CALLS = 12


class MissingAPIKey(RuntimeError):
    """Raised when a required API key is absent from the environment."""


@dataclass
class Settings:
    """Validated API keys for the current run."""

    gemini_api_key: str
    tavily_api_key: str


def load_settings() -> Settings:
    """Load API keys from the environment (.env supported) and validate them.

    Raises MissingAPIKey with a clear message if either key is missing, so the
    caller can fail fast instead of getting an opaque error deep in an API client.
    """
    load_dotenv()
    gemini_key = os.getenv("GEMINI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")

    missing = [
        name
        for name, value in (
            ("GEMINI_API_KEY", gemini_key),
            ("TAVILY_API_KEY", tavily_key),
        )
        if not value
    ]
    if missing:
        raise MissingAPIKey(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + ". Copy .env.example to .env and fill in your keys."
        )

    return Settings(gemini_api_key=gemini_key, tavily_api_key=tavily_key)
