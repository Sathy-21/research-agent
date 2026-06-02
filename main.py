"""CLI entry point for the research agent.

Usage:
    python main.py "your research question here"
    python main.py            # prompts for the question interactively
"""

from __future__ import annotations

import sys

from research_agent import config
from research_agent.agent import run_research


def main() -> int:
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        question = input("Enter your research question: ").strip()
    if not question:
        print("No research question provided.")
        return 1

    try:
        result = run_research(question)
    except config.MissingAPIKey as error:
        print(f"Configuration error: {error}")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
