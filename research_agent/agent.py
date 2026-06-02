"""Agent orchestration.

This module is the readable control-flow heart of the project. `run_research` reads
top to bottom as the whole agent:

    0. Validate  — normalize/guard the question before spending any API call (Phase 4).
    1. Plan      — decompose the question into sub-questions (Phase 2).
    2. Loop      — for each sub-question, run the per-sub-question pipeline:
                   search -> fetch + extract -> relevance filter -> grounded synthesis.
                   A sub-question that fails (after retries) or finds nothing relevant is
                   SKIPPED and the run continues (Phase 4 partial-failure resilience).
    3. Compose   — merge the surviving sub-answers into one narrative report.
    4. Verify    — check the report's claims against the retrieved source text, flag
                   unsupported ones, and compute a grounding metric (Phase 3).

Guardrails (Phase 4): a `Budget` caps total searches/LLM calls, a wall-clock deadline
caps total run time, and both degrade gracefully (compose what's done) rather than
crash. Transient network failures are retried by `retries.py`. Diagnostics go through
the `logging` module, and a concise run summary is logged at the end.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from groq import Groq
from tavily import TavilyClient

from . import compose, config, planner, relevance, retrieval, synthesis, verify
from .synthesis import AnsweredSubquestion
from .verify import VerificationReport

logger = logging.getLogger(__name__)


class InvalidQuestion(ValueError):
    """Raised when the research question fails input validation."""


class NoResultsError(RuntimeError):
    """Raised when no sub-question could be answered (the run has nothing to report)."""


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
class RunSummary:
    """Concise end-of-run metrics, logged at the end of a run."""

    llm_calls: int
    search_calls: int
    elapsed_seconds: float
    subquestions_total: int
    subquestions_succeeded: int
    subquestions_skipped: int
    percent_grounded: float


@dataclass
class ResearchResult:
    """Everything a caller needs to display the outcome of a run."""

    question: str
    subquestions: list[str]
    answered: list[AnsweredSubquestion]
    skipped: list[tuple[str, str]]  # (sub-question, reason it was dropped)
    report: str
    verification: VerificationReport
    summary: RunSummary
    cut_short: bool


def _validate_question(question: str) -> str:
    """Trim/normalize whitespace and reject empty or absurdly long input.

    Runs before any API call so bad input never costs a request.
    """
    normalized = " ".join(question.split())
    if not normalized:
        raise InvalidQuestion("The research question is empty.")
    if len(normalized) > config.MAX_QUESTION_CHARS:
        raise InvalidQuestion(
            f"The research question is too long ({len(normalized)} characters; "
            f"the limit is {config.MAX_QUESTION_CHARS})."
        )
    return normalized


def _research_one(
    llm_client: Groq,
    tavily_client: TavilyClient,
    subquestion: str,
    budget: Budget,
) -> AnsweredSubquestion | None:
    """Run search -> relevance filter -> synthesis for one sub-question.

    Returns the answer, or None if the sub-question should be skipped (no sources, or
    nothing relevant survived filtering). Raises if a step fails outright (after
    retries) so the caller can record it as a dropped sub-question.
    """
    logger.info("Sub-question: %s", subquestion)

    sources = retrieval.gather_sources(tavily_client, subquestion)
    budget.searches += 1
    if not sources:
        logger.info("  no sources retrieved; skipping")
        return None

    relevant = relevance.filter_sources(llm_client, subquestion, sources)
    budget.llm_calls += 1
    logger.info("  relevance filter kept %d of %d source(s)", len(relevant), len(sources))
    if not relevant:
        logger.info("  no relevant sources; skipping")
        return None

    answer = synthesis.answer_subquestion(llm_client, subquestion, relevant)
    budget.llm_calls += 1
    logger.info("  synthesized answer from %d source(s)", len(answer.sources))
    return answer


def run_research(question: str) -> ResearchResult:
    """Run the full research pipeline for a question and return the result."""
    started_at = time.monotonic()
    deadline = started_at + config.MAX_RUN_SECONDS

    # --- 0. Validate input before spending any API call ---
    question = _validate_question(question)

    settings = config.load_settings()
    llm_client = Groq(api_key=settings.groq_api_key)
    tavily_client = TavilyClient(api_key=settings.tavily_api_key)
    budget = Budget()

    # --- 1. Plan: decompose the question (fatal if it can't, via PlanningError) ---
    logger.info("Planning sub-questions for: %s", question)
    subquestions = planner.make_plan(llm_client, question)
    budget.llm_calls += 1
    logger.info("Plan: %d sub-question(s)", len(subquestions))

    # --- 2. Loop: research each sub-question, skipping any that fail ---
    answered: list[AnsweredSubquestion] = []
    skipped: list[tuple[str, str]] = []
    cut_short = False

    for index, subquestion in enumerate(subquestions):
        remaining = subquestions[index:]

        if time.monotonic() > deadline:
            logger.warning(
                "Time limit (%.0fs) reached; composing with what's done",
                config.MAX_RUN_SECONDS,
            )
            cut_short = True
            skipped.extend((sq, "run time limit reached") for sq in remaining)
            break

        if not budget.can_search() or not budget.can_call_llm():
            logger.warning(
                "Budget reached (searches %d/%d, LLM %d/%d); composing with what's done",
                budget.searches, budget.max_searches,
                budget.llm_calls, budget.max_llm_calls,
            )
            skipped.extend((sq, "budget cap reached") for sq in remaining)
            break

        try:
            answer = _research_one(llm_client, tavily_client, subquestion, budget)
        except Exception as exc:  # noqa: BLE001 - one bad sub-question must not kill the run
            logger.warning("  sub-question failed after retries; skipping: %r", exc)
            skipped.append((subquestion, f"failed after retries ({exc})"))
            continue

        if answer is None:
            skipped.append((subquestion, "no relevant sources found"))
            continue
        answered.append(answer)

    # Fatal only if nothing succeeded — otherwise report whatever we have.
    if not answered:
        raise NoResultsError(
            "No sub-questions could be answered (all failed or found no relevant sources)."
        )

    # --- 3. Compose: write the narrative body from the surviving sub-answers ---
    logger.info("Composing report from %d sub-answer(s)", len(answered))
    sources_used = compose.unique_sources(answered)
    report_body = compose.compose_report(llm_client, question, answered)
    budget.llm_calls += 1

    # --- 4. Verify: check claims against the retrieved source text ---
    logger.info("Verifying claims against %d source(s)", len(sources_used))
    verification = verify.verify_report(llm_client, report_body, sources_used)
    if sources_used:
        budget.llm_calls += 1  # verify only calls the model when there is text to check
    logger.info(
        "Verified %d claim(s): %d supported (%.0f%% grounded)",
        verification.total, verification.supported, verification.percent_grounded,
    )
    report_body = verify.flag_unsupported(report_body, verification)

    # Assemble the final report: narrative, then a coverage note, then the source list.
    report = _assemble_report(report_body, skipped, cut_short, sources_used)

    summary = RunSummary(
        llm_calls=budget.llm_calls,
        search_calls=budget.searches,
        elapsed_seconds=time.monotonic() - started_at,
        subquestions_total=len(subquestions),
        subquestions_succeeded=len(answered),
        subquestions_skipped=len(skipped),
        percent_grounded=verification.percent_grounded,
    )
    logger.info(
        "Run summary: %d LLM call(s), %d search(es), %.1fs, "
        "%d/%d sub-questions succeeded (%d skipped), %.0f%% grounded",
        summary.llm_calls, summary.search_calls, summary.elapsed_seconds,
        summary.subquestions_succeeded, summary.subquestions_total,
        summary.subquestions_skipped, summary.percent_grounded,
    )

    return ResearchResult(
        question=question,
        subquestions=subquestions,
        answered=answered,
        skipped=skipped,
        report=report,
        verification=verification,
        summary=summary,
        cut_short=cut_short,
    )


def _assemble_report(
    report_body: str,
    skipped: list[tuple[str, str]],
    cut_short: bool,
    sources_used: list[retrieval.Source],
) -> str:
    """Combine the narrative body, a coverage note, and the source list into one report."""
    sections = [report_body]

    note_lines: list[str] = []
    if cut_short:
        note_lines.append(
            "This run was cut short (time or budget limit), so the report may be incomplete."
        )
    if skipped:
        note_lines.append(
            "The following sub-questions were dropped and are NOT reflected above:"
        )
        note_lines.extend(f"- {sq} ({reason})" for sq, reason in skipped)
    if note_lines:
        sections.append("Coverage note\n-------------\n" + "\n".join(note_lines))

    source_list = compose.format_source_list(sources_used)
    if source_list:
        sections.append(source_list)

    return "\n\n".join(sections)
