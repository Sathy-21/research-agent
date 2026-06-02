"""Agent orchestration.

This module is the readable control-flow heart of the project. `run_research` reads
top to bottom as the whole agent:

    1. Plan    — decompose the question into sub-questions (Phase 2).
    2. Loop    — for each sub-question, run the Phase 1 pipeline:
                 search -> fetch + extract sources -> grounded synthesis.
    3. Compose — merge the sub-answers into one final report.

A `Budget` caps total searches and LLM calls so a run can't spiral in cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import anthropic
from tavily import TavilyClient

from . import compose, config, planner, retrieval, synthesis
from .synthesis import AnsweredSubquestion


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


def run_research(question: str) -> ResearchResult:
    """Run the full research pipeline for a question and return the result."""
    settings = config.load_settings()
    anthropic_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    tavily_client = TavilyClient(api_key=settings.tavily_api_key)
    budget = Budget()

    # --- 1. Plan: decompose the question into sub-questions (Phase 2) ---
    subquestions = planner.make_plan(anthropic_client, question)
    budget.llm_calls += 1

    # --- 2. Loop: run the Phase 1 pipeline for each sub-question ---
    answered: list[AnsweredSubquestion] = []
    for subquestion in subquestions:
        # Each iteration needs one search and one LLM call; stop launching new work
        # if either budget is exhausted and compose with what we have so far.
        if not budget.can_search() or not budget.can_call_llm():
            break
        sources = retrieval.gather_sources(tavily_client, subquestion)
        budget.searches += 1
        answer = synthesis.answer_subquestion(anthropic_client, subquestion, sources)
        budget.llm_calls += 1
        answered.append(answer)

    # --- 3. Compose: merge the sub-answers into one final report ---
    report = compose.compose_report(anthropic_client, question, answered)
    budget.llm_calls += 1

    return ResearchResult(
        question=question,
        subquestions=subquestions,
        answered=answered,
        report=report,
    )
