"""Configuration: API-key loading, model choices, and run limits.

All secrets are read from environment variables (loaded from a local .env file via
python-dotenv) — nothing is hardcoded. Call `load_settings()` to get validated keys.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# --- Model choices ---------------------------------------------------------------
# The user asked for cost-efficient models. We use Haiku for the high-volume steps
# (planning + per-sub-question synthesis). We step up to Sonnet for the single final
# compose call only: weaving many sub-answers into one coherent report is the most
# quality-sensitive step, and it runs exactly once per research run, so the extra
# cost is marginal while the quality gain is the most visible to the reader.
PLANNER_MODEL = "claude-haiku-4-5"
SYNTHESIS_MODEL = "claude-haiku-4-5"
COMPOSE_MODEL = "claude-sonnet-4-6"

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

    anthropic_api_key: str
    tavily_api_key: str


def load_settings() -> Settings:
    """Load API keys from the environment (.env supported) and validate them.

    Raises MissingAPIKey with a clear message if either key is missing, so the
    caller can fail fast instead of getting an opaque error deep in an API client.
    """
    load_dotenv()
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")

    missing = [
        name
        for name, value in (
            ("ANTHROPIC_API_KEY", anthropic_key),
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

    return Settings(anthropic_api_key=anthropic_key, tavily_api_key=tavily_key)
