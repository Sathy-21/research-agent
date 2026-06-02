"""Planning (Phase 2): decompose one research question into focused sub-questions.

A single LLM call turns the original question into 3-6 specific, non-overlapping
sub-questions, each answerable with one web search. If planning fails for any reason,
we fall back to researching the original question directly, so a run never dead-ends.
"""

from __future__ import annotations

from google import genai

from . import config, llm

_PLANNER_SYSTEM = (
    "You are a research planner. Given a single research question, break it into "
    f"{config.MIN_SUBQUESTIONS} to {config.MAX_SUBQUESTIONS} focused, non-overlapping "
    "sub-questions that together fully answer it. Each sub-question must be specific "
    "enough to answer with a single web search."
)


def make_plan(client: genai.Client, question: str) -> list[str]:
    """Decompose `question` into a list of sub-questions (one LLM call).

    Falls back to `[question]` if the model returns nothing usable.
    """
    user = (
        f"Research question:\n{question}\n\n"
        f"Return a JSON array of {config.MIN_SUBQUESTIONS}-{config.MAX_SUBQUESTIONS} "
        "sub-question strings, and nothing else."
    )

    try:
        data = llm.complete_json(
            client,
            model=config.PLANNER_MODEL,
            system=_PLANNER_SYSTEM,
            user=user,
            max_tokens=1024,
        )
        subquestions = [str(item).strip() for item in data if str(item).strip()]
    except (ValueError, TypeError):
        # Malformed JSON, or a non-iterable response — fall back below.
        subquestions = []

    if not subquestions:
        return [question]
    # Defensively enforce the cap in case the model overshoots.
    return subquestions[: config.MAX_SUBQUESTIONS]
