"""Composition: merge the sub-answers into one report, plus source-list helpers.

`compose_report` writes only the narrative body. The numbered source list is built
separately (`format_source_list`) and `unique_sources` is exposed so the verifier can
reach the underlying source text. Keeping the narrative separate from the source list
lets claim verification run on the prose alone, before the source list is attached.
"""

from __future__ import annotations

from groq import Groq

from . import config, llm
from .retrieval import Source
from .synthesis import AnsweredSubquestion

_COMPOSE_SYSTEM = (
    "You are a research writer. You will be given an original research question and a "
    "set of findings, each answering a sub-question. Write a single cohesive report "
    "that answers the original question, integrating the findings into a clear "
    "narrative. Base the report only on the findings provided; do not invent facts. "
    "Do not include a sources or references section — that is appended separately."
)


def unique_sources(answered: list[AnsweredSubquestion]) -> list[Source]:
    """Collect the unique sources used across all sub-answers, preserving order."""
    seen: set[str] = set()
    ordered: list[Source] = []
    for item in answered:
        for source in item.sources:
            if source.url not in seen:
                seen.add(source.url)
                ordered.append(source)
    return ordered


def compose_report(
    client: Groq, question: str, answered: list[AnsweredSubquestion]
) -> str:
    """Compose the narrative report body from the sub-answers (one LLM call).

    Returns a plain message without calling the model if there are no sub-answers
    (e.g. every sub-question was skipped for lack of relevant sources).
    """
    if not answered:
        return "No sufficiently relevant sources were found to answer this question."

    findings = "\n\n".join(
        f"Sub-question: {item.subquestion}\nFindings: {item.answer}"
        for item in answered
    )
    user = (
        f"Original research question:\n{question}\n\n"
        f"Findings from research:\n\n{findings}"
    )

    return llm.complete_text(
        client,
        model=config.COMPOSE_MODEL,
        system=_COMPOSE_SYSTEM,
        user=user,
        max_tokens=4000,
    )


def format_source_list(sources: list[Source]) -> str:
    """Render a deduped, numbered source list. Empty string if there are none."""
    if not sources:
        return ""
    lines = "\n".join(
        f"{i}. {source.title} — {source.url}"
        for i, source in enumerate(sources, start=1)
    )
    return f"Sources\n-------\n{lines}"
