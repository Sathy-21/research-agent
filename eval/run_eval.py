"""Evaluation harness: run the agent over a benchmark and record process metrics.

Runs the full research agent on each benchmark question (sequentially, reusing the
agent's retry/backoff and guardrails) and records per question: grounding %, claim
counts (total / supported / fabricated), sub-questions succeeded vs skipped, LLM calls,
search calls, and elapsed time. Raw results are saved to eval/results/<timestamp>.json
and a summary table (per-question rows + aggregate means) is printed to stdout.

What this measures: grounding and process metrics — how well the report's claims are
supported by retrieved sources, and how the pipeline behaved. It does NOT measure the
factual correctness of the answers (there are no gold answers).

Usage:
    python eval/run_eval.py                 # full benchmark
    python eval/run_eval.py --limit 3       # first 3 questions (quick smoke test)
    python eval/run_eval.py --delay 8       # seconds to wait between questions
    VERIFIER_MODE=old python eval/run_eval.py   # run with the old extraction prompt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

# Make the project package importable when run as a script from the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from research_agent import config  # noqa: E402
from research_agent.agent import (  # noqa: E402
    InvalidQuestion,
    NoResultsError,
    run_research,
)
from research_agent.planner import PlanningError  # noqa: E402

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_QUESTIONS_PATH = os.path.join(_EVAL_DIR, "questions.json")
_RESULTS_DIR = os.path.join(_EVAL_DIR, "results")

logger = logging.getLogger("eval")

# Per-question fields that are numeric and averaged in the aggregate.
_NUMERIC_FIELDS = [
    "grounding_pct",
    "total_claims",
    "supported_claims",
    "fabricated_claims",
    "subq_succeeded",
    "subq_skipped",
    "llm_calls",
    "search_calls",
    "elapsed_seconds",
]


def load_questions(limit: int | None) -> list[dict]:
    with open(_QUESTIONS_PATH, encoding="utf-8") as handle:
        questions = json.load(handle)
    return questions[:limit] if limit else questions


def run_one(entry: dict) -> dict:
    """Run the agent on one benchmark question and extract its metrics."""
    qid, question = entry["id"], entry["question"]
    logger.info("[%s] %s", qid, question)
    try:
        result = run_research(question)
    except (PlanningError, NoResultsError, InvalidQuestion) as error:
        logger.warning("[%s] failed: %s", qid, error)
        return {"id": qid, "question": question, "status": "error",
                "error": f"{type(error).__name__}: {error}"}

    verification, summary = result.verification, result.summary
    return {
        "id": qid,
        "question": question,
        "status": "ok",
        "grounding_pct": round(verification.percent_grounded, 1),
        "total_claims": verification.total,
        "supported_claims": verification.supported,
        "fabricated_claims": verification.fabricated,
        "subq_succeeded": summary.subquestions_succeeded,
        "subq_skipped": summary.subquestions_skipped,
        "llm_calls": summary.llm_calls,
        "search_calls": summary.search_calls,
        "elapsed_seconds": round(summary.elapsed_seconds, 1),
    }


def aggregate(rows: list[dict]) -> dict:
    """Mean of each numeric field over the questions that completed successfully."""
    ok = [r for r in rows if r["status"] == "ok"]
    means = {}
    for field in _NUMERIC_FIELDS:
        means[f"mean_{field}"] = round(sum(r[field] for r in ok) / len(ok), 1) if ok else None
    return {"n_total": len(rows), "n_ok": len(ok), "n_error": len(rows) - len(ok), **means}


def print_table(rows: list[dict], agg: dict) -> None:
    """Print a per-question table plus aggregate means to stdout."""
    header = f"{'id':<5} {'status':<6} {'grnd%':>6} {'clm':>4} {'sup':>4} {'fab':>4} " \
             f"{'sq+':>4} {'sq-':>4} {'llm':>4} {'srch':>5} {'sec':>6}"
    print(header)
    print("-" * len(header))
    for r in rows:
        if r["status"] != "ok":
            print(f"{r['id']:<5} {'ERROR':<6}  (see results JSON: {r.get('error','')[:40]})")
            continue
        print(f"{r['id']:<5} {'ok':<6} {r['grounding_pct']:>6} {r['total_claims']:>4} "
              f"{r['supported_claims']:>4} {r['fabricated_claims']:>4} {r['subq_succeeded']:>4} "
              f"{r['subq_skipped']:>4} {r['llm_calls']:>4} {r['search_calls']:>5} "
              f"{r['elapsed_seconds']:>6}")
    print("-" * len(header))
    print(f"Questions: {agg['n_ok']} ok / {agg['n_error']} error / {agg['n_total']} total")
    if agg["n_ok"]:
        print(f"Mean grounding:   {agg['mean_grounding_pct']}%")
        print(f"Mean claims/run:  {agg['mean_total_claims']} "
              f"(supported {agg['mean_supported_claims']}, fabricated {agg['mean_fabricated_claims']})")
        print(f"Mean sub-questions succeeded/skipped: "
              f"{agg['mean_subq_succeeded']} / {agg['mean_subq_skipped']}")
        print(f"Mean LLM calls: {agg['mean_llm_calls']}   mean searches: {agg['mean_search_calls']}"
              f"   mean seconds: {agg['mean_elapsed_seconds']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the research agent over a benchmark.")
    parser.add_argument("--limit", type=int, default=None,
                        help="run only the first N questions (for a quick smoke test)")
    parser.add_argument("--delay", type=float, default=5.0,
                        help="seconds to wait between questions (free-tier rate-limit guard)")
    parser.add_argument("--mode", choices=["old", "new"], default=None,
                        help="verifier mode for this run (overrides VERIFIER_MODE env; "
                             "defaults to the env var, or 'new')")
    parser.add_argument("--verbose", action="store_true", help="enable DEBUG logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    # --mode wins if passed; otherwise fall back to the env var (default "new").
    verifier_mode = (args.mode or os.getenv("VERIFIER_MODE", "new")).strip().lower()
    # Export it so the agent's verifier (config.verifier_mode()) picks up this run's mode.
    os.environ["VERIFIER_MODE"] = verifier_mode
    questions = load_questions(args.limit)
    logger.info("Running %d question(s), verifier_mode=%s, delay=%.0fs",
                len(questions), verifier_mode, args.delay)

    rows: list[dict] = []
    try:
        for index, entry in enumerate(questions):
            rows.append(run_one(entry))
            # Pause between questions (not after the last) to respect per-minute limits.
            if index < len(questions) - 1 and args.delay > 0:
                time.sleep(args.delay)
    except config.MissingAPIKey as error:
        print(f"Configuration error: {error}")
        return 1

    agg = aggregate(rows)
    output = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "verifier_mode": verifier_mode,
        "delay_seconds": args.delay,
        "n_questions": len(questions),
        "questions": rows,
        "aggregate": agg,
    }

    os.makedirs(_RESULTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(_RESULTS_DIR, f"{verifier_mode}-{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print()
    print_table(rows, agg)
    print(f"\nRaw results saved to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
