"""Relevance filtering (Phase 3): drop off-topic sources before synthesis.

Defends against a real failure: a sub-question's search pulled in off-topic sources
(vision/multimodal papers for a question about text-retrieval systems) and the
synthesizer summarized them anyway, because nothing checked relevance. Before
synthesizing, we ask the model — in ONE batched call per sub-question, not one call per
source — which of the gathered sources are actually relevant, and keep only those.
"""

from __future__ import annotations

from groq import Groq

from . import config, llm
from .retrieval import Source

# Only a short snippet of each source is needed to judge relevance; keeping it small
# keeps the single batched call cheap and well under the token budget.
_SNIPPET_CHARS = 500

_RELEVANCE_SYSTEM = (
    "You decide which sources are relevant to a specific sub-question. A source is "
    "relevant only if its content could genuinely help answer that sub-question. Sources "
    "on a different subject or domain are NOT relevant, even if superficially similar."
)


def filter_sources(
    client: Groq, subquestion: str, sources: list[Source]
) -> list[Source]:
    """Return only the sources relevant to `subquestion` (one batched LLM call).

    If the model's judgment can't be parsed, conservatively keep all sources so the
    pipeline degrades to pre-filter behaviour rather than discarding everything. An
    empty result (the model judged none relevant) is returned as-is so the caller can
    skip the sub-question.
    """
    if not sources:
        return []

    listing = "\n\n".join(
        f"[{i}] {source.title}\n{source.text[:_SNIPPET_CHARS]}"
        for i, source in enumerate(sources, start=1)
    )
    user = (
        f"Sub-question:\n{subquestion}\n\n"
        f"Sources:\n{listing}\n\n"
        'Respond with JSON only: {"relevant_ids": [<the numbers of the relevant sources>]}'
    )

    try:
        data = llm.complete_json(
            client,
            model=config.RELEVANCE_MODEL,
            system=_RELEVANCE_SYSTEM,
            user=user,
            max_tokens=512,
        )
        ids = data.get("relevant_ids", [])
        return [
            sources[i - 1]
            for i in ids
            if isinstance(i, int) and 1 <= i <= len(sources)
        ]
    except (ValueError, AttributeError):
        # Malformed JSON or unexpected shape — keep everything rather than crash.
        return list(sources)
