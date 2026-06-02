"""CLI entry point for the research agent.

Usage:
    python main.py "your research question here"
    python main.py --verbose "..."     # turn up diagnostic logging (DEBUG)
    python main.py                      # prompts for the question interactively

The user-facing report is printed to stdout. All diagnostic detail (phases, retries,
per-sub-question progress, the run summary) goes through `logging` to stderr, so stdout
stays clean and the noise can be turned up (--verbose / LOG_LEVEL) or down.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from research_agent import config, verify
from research_agent.agent import InvalidQuestion, NoResultsError, run_research
from research_agent.planner import PlanningError


def _configure_logging(verbose: bool) -> None:
    """Set up logging. Level: --verbose -> DEBUG, else LOG_LEVEL env, else INFO."""
    level_name = "DEBUG" if verbose else os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, level_name.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Autonomous research agent.")
    parser.add_argument("question", nargs="*", help="the research question")
    parser.add_argument("--verbose", action="store_true", help="enable DEBUG logging")
    args = parser.parse_args()

    _configure_logging(args.verbose)

    question = " ".join(args.question).strip()
    if not question:
        question = input("Enter your research question: ").strip()

    try:
        result = run_research(question)
    except config.MissingAPIKey as error:
        print(f"Configuration error: {error}")
        return 1
    except InvalidQuestion as error:
        print(f"Invalid question: {error}")
        return 1
    except PlanningError as error:
        print(f"Planning failed: {error}")
        return 1
    except NoResultsError as error:
        print(f"No results: {error}")
        return 1

    separator = "=" * 70
    print(separator)
    print("RESEARCH QUESTION")
    print(separator)
    print(result.question)

    print("\n" + separator)
    print("SUB-QUESTIONS")
    print(separator)
    for i, subquestion in enumerate(result.subquestions, start=1):
        print(f"  {i}. {subquestion}")

    print("\n" + separator)
    print("FINAL REPORT")
    print(separator)
    print(result.report)

    print("\n" + separator)
    print("GROUNDING")
    print(separator)
    print(verify.grounding_summary(result.verification))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
