"""Search & retrieval (Phase 1, steps 1-2): web search, then readable-text extraction.

`gather_sources` runs one Tavily web search and then fetches and extracts the readable
text of each top hit with trafilatura. Any source that fails to load is skipped so the
agent keeps going instead of crashing on a single bad page.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import trafilatura
from tavily import TavilyClient

from . import config
from .retries import call_with_retries

logger = logging.getLogger(__name__)


@dataclass
class Source:
    """A single retrieved source with its extracted readable text."""

    title: str
    url: str
    text: str


def search_hits(tavily: TavilyClient, query: str) -> list[dict]:
    """Run one web search and return the top result hits (each a dict with url/title).

    The search call is retried on transient failures by the shared retry layer. A
    permanent error or an exhausted retry propagates, so the caller can skip the
    sub-question; an empty result set (no hits) is returned normally.
    """
    response = call_with_retries(
        lambda: tavily.search(query=query, max_results=config.RESULTS_PER_SUBQUESTION),
        description=f"Tavily search ({query!r})",
    )
    hits = response.get("results", [])
    logger.debug("Search for %r returned %d hit(s)", query, len(hits))
    return hits


def fetch_source(hit: dict) -> Source | None:
    """Fetch one URL and extract its readable text.

    Returns None on any failure (no URL, page won't load, nothing extractable) so the
    caller can skip this source and continue.
    """
    url = hit.get("url")
    if not url:
        return None
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        text = trafilatura.extract(downloaded)
    except Exception:
        return None
    if not text:
        return None
    return Source(
        title=hit.get("title") or url,
        url=url,
        text=text[: config.MAX_SOURCE_CHARS],
    )


def gather_sources(tavily: TavilyClient, query: str) -> list[Source]:
    """Search for `query`, then fetch + extract each top hit, skipping failures."""
    hits = search_hits(tavily, query)
    sources: list[Source] = []
    for hit in hits:
        source = fetch_source(hit)
        if source is not None:
            sources.append(source)
    skipped = len(hits) - len(sources)
    if skipped:
        logger.debug("%d of %d page(s) could not be fetched/extracted", skipped, len(hits))
    return sources
