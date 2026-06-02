"""Composition: merge the sub-answers into one report and append a source list.

The narrative report is written by the stronger compose model from the collected
sub-answers. The numbered source list is built deterministically in code (deduped,
order-preserving) rather than asked of the model, so it is always accurate.
"""

from __future__ import annotations

import anthropic

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


def _unique_sources(answered: list[AnsweredSubquestion]) -> list[Source]:
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
    client: anthropic.Anthropic, question: str, answered: list[AnsweredSubquestion]
) -> str:
    """Compose the final report (one LLM call) and append a numbered source list."""
    findings = "\n\n".join(
        f"Sub-question: {item.subquestion}\nFindings: {item.answer}"
        for item in answered
    )
    user = (
        f"Original research question:\n{question}\n\n"
        f"Findings from research:\n\n{findings}"
    )

    body = llm.complete_text(
        client,
        model=config.COMPOSE_MODEL,
        system=_COMPOSE_SYSTEM,
        user=user,
        max_tokens=4000,
    )

    sources = _unique_sources(answered)
    if not sources:
        return body

    source_lines = "\n".join(
        f"{i}. {source.title} — {source.url}"
        for i, source in enumerate(sources, start=1)
    )
    return f"{body}\n\nSources\n-------\n{source_lines}"
