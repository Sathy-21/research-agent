"""Configuration: API-key loading, model choices, and run limits.

All secrets are read from environment variables (loaded from a local .env file via
python-dotenv) — nothing is hardcoded. Call `load_settings()` to get validated keys.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# --- Model choices ---------------------------------------------------------------
# We use Groq's free tier, which offers a much larger daily request allowance than
# Gemini's free tier (enough to actually iterate on this multi-call agent). We use
# llama-3.3-70b-versatile for every step (planning, relevance filtering, synthesis,
# compose, verification): it is a strong general model that comfortably handles all of
# them on the free tier. The constants are kept separate so a stronger model can be
# dropped into any single step later without touching the rest of the code.
PLANNER_MODEL = "llama-3.3-70b-versatile"
RELEVANCE_MODEL = "llama-3.3-70b-versatile"
SYNTHESIS_MODEL = "llama-3.3-70b-versatile"
COMPOSE_MODEL = "llama-3.3-70b-versatile"
VERIFY_MODEL = "llama-3.3-70b-versatile"

# --- Pipeline knobs --------------------------------------------------------------
MIN_SUBQUESTIONS = 3
MAX_SUBQUESTIONS = 6
RESULTS_PER_SUBQUESTION = 3   # fetch + extract the top N web-search hits per sub-question
MAX_SOURCE_CHARS = 6000       # truncate each extracted page to bound token cost

# --- Cost guardrails -------------------------------------------------------------
# Hard ceilings so a single run can never spiral in cost, regardless of how many
# sub-questions the planner returns or how the loop behaves.
# Worst case (6 sub-questions): 1 plan + 6*(1 relevance + 1 synthesis) + 1 compose
#   + 1 verification = 15 LLM calls, so the cap is set just above that.
MAX_SEARCHES = 8
MAX_LLM_CALLS = 16


class MissingAPIKey(RuntimeError):
    """Raised when a required API key is absent from the environment."""


@dataclass
class Settings:
    """Validated API keys for the current run."""

    groq_api_key: str
    tavily_api_key: str


def load_settings() -> Settings:
    """Load API keys from the environment (.env supported) and validate them.

    Raises MissingAPIKey with a clear message if either key is missing, so the
    caller can fail fast instead of getting an opaque error deep in an API client.
    """
    load_dotenv()
    groq_key = os.getenv("GROQ_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")

    missing = [
        name
        for name, value in (
            ("GROQ_API_KEY", groq_key),
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

    return Settings(groq_api_key=groq_key, tavily_api_key=tavily_key)
