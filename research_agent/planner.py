"""Planning (Phase 2): decompose one research question into focused sub-questions.

A single LLM call turns the original question into 3-6 specific, non-overlapping
sub-questions, each answerable with one web search. The reply is parsed defensively:
the list of sub-questions is extracted whether the model returns a bare JSON array or
an object that wraps the array under a key. An invalid plan triggers one retry and then
a clear failure, so the agent never proceeds on garbage.
"""

from __future__ import annotations

import logging

from groq import Groq

from . import config, llm

logger = logging.getLogger(__name__)

# A real plan has at least this many sub-questions; fewer means the reply was a parsing
# artifact (e.g. a lone wrapper-key string), not a usable plan.
_MIN_VALID_SUBQUESTIONS = 2
_MAX_ATTEMPTS = 2  # one initial attempt plus one retry

_PLANNER_SYSTEM = (
    "You are a research planner. Given a single research question, break it into "
    f"{config.MIN_SUBQUESTIONS} to {config.MAX_SUBQUESTIONS} focused, non-overlapping "
    "sub-questions that together fully answer it. Each sub-question must be specific "
    "enough to answer with a single web search."
)


class PlanningError(RuntimeError):
    """Raised when the planner cannot produce a usable set of sub-questions."""


def _looks_like_key_name(text: str) -> bool:
    """True if `text` looks like a JSON wrapper key rather than a question.

    A genuine sub-question is a sentence containing spaces; a bare token such as
    "sub_questions" is almost certainly the object key leaking through, not content.
    """
    return " " not in text


def _parse_subquestions(data: object) -> list[str]:
    """Extract clean sub-question strings from a parsed JSON reply (any list shape)."""
    items = llm.extract_list(data)
    return [str(item).strip() for item in items if str(item).strip()]


def _is_valid_plan(subquestions: list[str]) -> bool:
    """Reject empties, too-short plans, and lone wrapper-key artifacts."""
    if len(subquestions) < _MIN_VALID_SUBQUESTIONS:
        return False
    if len(subquestions) == 1 and _looks_like_key_name(subquestions[0]):
        return False
    return True


def make_plan(client: Groq, question: str) -> list[str]:
    """Decompose `question` into a list of sub-questions.

    Tries up to twice (one retry). Raises PlanningError if no usable plan comes back,
    rather than proceeding with garbage.
    """
    user = (
        f"Research question:\n{question}\n\n"
        f"Return {config.MIN_SUBQUESTIONS}-{config.MAX_SUBQUESTIONS} sub-question strings "
        'as JSON. A bare array is fine, or an object like {"sub_questions": [...]}.'
    )

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            data = llm.complete_json(
                client,
                model=config.PLANNER_MODEL,
                system=_PLANNER_SYSTEM,
                user=user,
                max_tokens=1024,
            )
        except ValueError:
            logger.warning("Planner returned unparseable JSON (attempt %d/%d)", attempt, _MAX_ATTEMPTS)
            continue  # unparseable JSON — retry

        subquestions = _parse_subquestions(data)
        if _is_valid_plan(subquestions):
            # Defensively enforce the cap in case the model overshoots.
            return subquestions[: config.MAX_SUBQUESTIONS]
        logger.warning(
            "Planner produced an invalid plan (attempt %d/%d): %r",
            attempt, _MAX_ATTEMPTS, subquestions,
        )

    raise PlanningError(
        "Couldn't generate sub-questions. This can happen with a temporary model issue "
        "(rerunning may help) or if the input isn't a researchable question."
    )
