"""Synthesis (Phase 1, step 3): write a short, grounded answer to one sub-question.

The model is given the sub-question plus the numbered extracted sources, and must
answer using only those sources. It also reports which source numbers it drew from,
so we can track the exact URLs behind each sub-answer for the final source list.
"""

from __future__ import annotations

from dataclasses import dataclass

from google import genai

from . import config, llm
from .retrieval import Source


@dataclass
class AnsweredSubquestion:
    """A sub-question, its grounded answer, and the sources the answer actually used."""

    subquestion: str
    answer: str
    sources: list[Source]


_SYNTHESIS_SYSTEM = (
    "You are a careful research assistant. Answer the sub-question using ONLY the "
    "provided sources. Be concise (a short paragraph). Do not invent facts that are "
    "not supported by the sources. If the sources do not answer the sub-question, say "
    "so plainly."
)


def answer_subquestion(
    client: genai.Client, subquestion: str, sources: list[Source]
) -> AnsweredSubquestion:
    """Produce a grounded answer to `subquestion` from `sources` (one LLM call).

    If no sources were retrieved, returns a placeholder answer without calling the
    model (so we don't waste an LLM call on an empty context).
    """
    if not sources:
        return AnsweredSubquestion(
            subquestion,
            "No sources could be retrieved for this sub-question.",
            [],
        )

    numbered = "\n\n".join(
        f"[{i}] {source.title} ({source.url})\n{source.text}"
        for i, source in enumerate(sources, start=1)
    )
    user = (
        f"Sub-question:\n{subquestion}\n\n"
        f"Sources:\n{numbered}\n\n"
        'Respond with JSON only, in this exact shape:\n'
        '{"answer": "<your grounded answer>", '
        '"sources_used": [<the source numbers you drew from>]}'
    )

    try:
        data = llm.complete_json(
            client,
            model=config.SYNTHESIS_MODEL,
            system=_SYNTHESIS_SYSTEM,
            user=user,
            max_tokens=1024,
        )
        answer = str(data.get("answer", "")).strip()
        numbers = data.get("sources_used", []) or []
        used = [
            sources[n - 1]
            for n in numbers
            if isinstance(n, int) and 1 <= n <= len(sources)
        ]
    except (ValueError, AttributeError):
        # Malformed JSON or unexpected shape — degrade gracefully.
        answer, used = "", []

    if not answer:
        return AnsweredSubquestion(
            subquestion,
            "Could not synthesize an answer for this sub-question.",
            [],
        )
    return AnsweredSubquestion(subquestion, answer, used)
