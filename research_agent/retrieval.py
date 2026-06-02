"""Search & retrieval (Phase 1, steps 1-2): web search, then readable-text extraction.

`gather_sources` runs one Tavily web search and then fetches and extracts the readable
text of each top hit with trafilatura. Any source that fails to load is skipped so the
agent keeps going instead of crashing on a single bad page.
"""

from __future__ import annotations

from dataclasses import dataclass

import trafilatura
from tavily import TavilyClient

from . import config


@dataclass
class Source:
    """A single retrieved source with its extracted readable text."""

    title: str
    url: str
    text: str


def search_hits(tavily: TavilyClient, query: str) -> list[dict]:
    """Run one web search and return the top result hits (each a dict with url/title).

    Returns an empty list on failure so a sub-question with no results degrades
    gracefully rather than raising.
    """
    try:
        response = tavily.search(query=query, max_results=config.RESULTS_PER_SUBQUESTION)
    except Exception:
        return []
    return response.get("results", [])


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
    sources: list[Source] = []
    for hit in search_hits(tavily, query):
        source = fetch_source(hit)
        if source is not None:
            sources.append(source)
    return sources
