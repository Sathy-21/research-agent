"""Agent orchestration.

This module is the readable control-flow heart of the project. `run_research` reads
top to bottom as the whole agent:

    1. Plan      — decompose the question into sub-questions (Phase 2).
    2. Loop      — for each sub-question, run the per-sub-question pipeline:
                   search -> fetch + extract -> relevance filter -> grounded synthesis.
    3. Compose   — merge the sub-answers into one narrative report.
    4. Verify    — check the report's claims against the retrieved source text,
                   flag unsupported ones, and compute a grounding metric (Phase 3).

A `Budget` caps total searches and LLM calls so a run can't spiral in cost.
"""

from __future__ import annotations

from dataclasses import dataclass

from groq import Groq
from tavily import TavilyClient

from . import compose, config, planner, relevance, retrieval, synthesis, verify
from .synthesis import AnsweredSubquestion
from .verify import VerificationReport


@dataclass
class Budget:
    """Hard caps so a single run can't spiral in cost.

    The orchestrator records each search and LLM call here and checks the budget
    before launching new work.
    """

    max_searches: int = config.MAX_SEARCHES
    max_llm_calls: int = config.MAX_LLM_CALLS
    searches: int = 0
    llm_calls: int = 0

    def can_search(self) -> bool:
        return self.searches < self.max_searches

    def can_call_llm(self) -> bool:
        return self.llm_calls < self.max_llm_calls


@dataclass
class ResearchResult:
    """Everything a caller needs to display the outcome of a run."""

    question: str
    subquestions: list[str]
    answered: list[AnsweredSubquestion]
    report: str
    verification: VerificationReport


def run_research(question: str) -> ResearchResult:
    """Run the full research pipeline for a question and return the result."""
    settings = config.load_settings()
    llm_client = Groq(api_key=settings.groq_api_key)
    tavily_client = TavilyClient(api_key=settings.tavily_api_key)
    budget = Budget()

    # --- 1. Plan: decompose the question into sub-questions (Phase 2) ---
    subquestions = planner.make_plan(llm_client, question)
    budget.llm_calls += 1

    # --- 2. Loop: per-sub-question search -> filter -> synthesis ---
    answered: list[AnsweredSubquestion] = []
    for subquestion in subquestions:
        # Each iteration needs a search and up to two LLM calls (relevance + synthesis);
        # stop launching new work once either budget is exhausted and compose what we have.
        if not budget.can_search() or not budget.can_call_llm():
            break

        sources = retrieval.gather_sources(tavily_client, subquestion)
        budget.searches += 1
        if not sources:
            continue  # no results for this sub-question; skip gracefully

        # Phase 3A: drop off-topic sources in one batched call.
        relevant = relevance.filter_sources(llm_client, subquestion, sources)
        budget.llm_calls += 1
        if not relevant:
            continue  # nothing on-topic survived; skip rather than synthesize from nothing

        if not budget.can_call_llm():
            break
        answer = synthesis.answer_subquestion(llm_client, subquestion, relevant)
        budget.llm_calls += 1
        answered.append(answer)

    # --- 3. Compose: write the narrative body from the sub-answers ---
    sources_used = compose.unique_sources(answered)
    report_body = compose.compose_report(llm_client, question, answered)
    if answered:
        budget.llm_calls += 1  # compose only calls the model when there are sub-answers

    # --- 4. Verify: check claims against the retrieved source text (Phase 3B) ---
    verification = verify.verify_report(llm_client, report_body, sources_used)
    if sources_used:
        budget.llm_calls += 1  # verify only calls the model when there is text to check
    report_body = verify.flag_unsupported(report_body, verification)

    # Attach the deduped source list to the (now flagged) narrative.
    source_list = compose.format_source_list(sources_used)
    report = f"{report_body}\n\n{source_list}" if source_list else report_body

    return ResearchResult(
        question=question,
        subquestions=subquestions,
        answered=answered,
        report=report,
        verification=verification,
    )
