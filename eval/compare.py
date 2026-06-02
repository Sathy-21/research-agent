"""Compare two eval result files (e.g. old vs new verifier extraction).

Prints mean grounding before vs after, mean claims, and how many fabricated/non-report
claims each run produced (the `fabricated_claims` metric — claims the verifier extracted
that do not actually appear in the report, per the word-overlap heuristic in verify.py).

Usage:
    python eval/compare.py eval/results/old-<ts>.json eval/results/new-<ts>.json
"""

from __future__ import annotations

import argparse
import json


def _summarize(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    ok = [q for q in data["questions"] if q["status"] == "ok"]
    agg = data.get("aggregate", {})
    return {
        "label": data.get("verifier_mode", path),
        "n_ok": len(ok),
        "mean_grounding": agg.get("mean_grounding_pct"),
        "mean_claims": agg.get("mean_total_claims"),
        "mean_fabricated": agg.get("mean_fabricated_claims"),
        "total_fabricated": sum(q["fabricated_claims"] for q in ok),
        "total_claims": sum(q["total_claims"] for q in ok),
        "by_id": {q["id"]: q for q in ok},
    }


def _fmt(value: object) -> str:
    return "n/a" if value is None else str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two eval result JSON files.")
    parser.add_argument("before", help="results JSON for the 'before' run (e.g. old mode)")
    parser.add_argument("after", help="results JSON for the 'after' run (e.g. new mode)")
    args = parser.parse_args()

    before, after = _summarize(args.before), _summarize(args.after)
    b, a = before["label"], after["label"]

    print(f"{'metric':<28} {b + ' (before)':>16} {a + ' (after)':>16}")
    print("-" * 62)
    rows = [
        ("questions ok", before["n_ok"], after["n_ok"]),
        ("mean grounding %", before["mean_grounding"], after["mean_grounding"]),
        ("mean claims / run", before["mean_claims"], after["mean_claims"]),
        ("mean fabricated / run", before["mean_fabricated"], after["mean_fabricated"]),
        ("total fabricated claims", before["total_fabricated"], after["total_fabricated"]),
        ("total claims", before["total_claims"], after["total_claims"]),
    ]
    for name, bv, av in rows:
        print(f"{name:<28} {_fmt(bv):>16} {_fmt(av):>16}")

    # Per-question grounding side by side, for ids present in both runs.
    shared = sorted(set(before["by_id"]) & set(after["by_id"]))
    if shared:
        print(f"\n{'id':<6} {'before%':>8} {'after%':>8} {'delta':>8}")
        print("-" * 32)
        for qid in shared:
            bp = before["by_id"][qid]["grounding_pct"]
            ap = after["by_id"][qid]["grounding_pct"]
            print(f"{qid:<6} {bp:>8} {ap:>8} {round(ap - bp, 1):>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
