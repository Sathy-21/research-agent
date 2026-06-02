"""Claim verification (Phase 3, hardened in Phase 5): check the report against sources.

After the report is composed, we extract its factual claims and check each one against
the ACTUAL retrieved source text — in ONE batched call, not one per claim — then report
a grounding metric.

Phase 5 fix — trustworthy extraction. The grounding number is only meaningful if the
claims being checked are claims the report actually makes. Earlier the extraction step
sometimes fabricated strawman claims (assertions the report never stated) and marked
them unsupported, distorting the metric. The extraction prompt is now strict: extract
only the report's own assertions, faithfully paraphrased, each a single checkable
statement, at a consistent granularity — never invented, generalized, or negated. As an
always-on instrument, each extracted claim is also checked (by a deterministic
word-overlap heuristic) for whether it genuinely appears in the report, so fabricated
claims are countable rather than silently skewing the grounding percentage.

We *flag* unsupported claims rather than deleting them, to keep the report intact and
transparent about what could not be grounded.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from groq import Groq

from . import config, llm
from .retrieval import Source

logger = logging.getLogger(__name__)

# Strict extraction + support check (default "new" mode). The extraction half is
# deliberately emphatic because that was the unreliable step (see DECISIONS.md).
_VERIFY_SYSTEM_NEW = (
    "You are a strict fact-checker working in two steps.\n"
    "STEP 1 — EXTRACT the report's own factual claims. Extract ONLY assertions the "
    "report actually makes, faithfully paraphrased (verbatim-ish), each a single, "
    "self-contained, checkable statement. Use a consistent granularity: every distinct "
    "factual assertion is its own claim — including specific named techniques, methods, "
    "numbers, and entities, not just broad summary sentences. Do NOT invent, generalize "
    "beyond, negate, combine, or otherwise add claims the report does not state. If the "
    "report makes no factual claims, return an empty list.\n"
    "STEP 2 — For each extracted claim, decide whether the SOURCE EXCERPTS support it. "
    "Mark it 'supported' only if the sources directly back it up; if the sources do not "
    "mention it, mark it unsupported. Judge support ONLY against the provided sources, "
    "never against prior knowledge."
)

# Original prompt, kept available behind VERIFIER_MODE=old for before/after comparison.
# This is the version whose loose extraction produced fabricated/strawman claims.
_VERIFY_SYSTEM_OLD = (
    "You are a strict fact-checker. You are given source excerpts and a report. Break the "
    "report into its distinct factual claims. For each claim, decide whether it is "
    "supported by the source excerpts: a claim is 'supported' only if the sources "
    "directly back it up. If the sources do not mention it, mark it unsupported. Judge "
    "ONLY against the provided sources, never against prior knowledge."
)

# Trailing user-message directive paired with each system prompt.
_USER_DIRECTIVE_NEW = (
    "Extract the factual claims the Report above actually makes (do not introduce any "
    "claim it does not state), then check each against the Sources. "
    'Respond with JSON only: '
    '{"claims": [{"claim": "<claim text>", "supported": true|false}, ...]}'
)
_USER_DIRECTIVE_OLD = (
    'Respond with JSON only: '
    '{"claims": [{"claim": "<claim text>", "supported": true|false}, ...]}'
)


def _select_prompt() -> tuple[str, str]:
    """Return (system, user_directive) for the active VERIFIER_MODE."""
    if config.verifier_mode() == "old":
        return _VERIFY_SYSTEM_OLD, _USER_DIRECTIVE_OLD
    return _VERIFY_SYSTEM_NEW, _USER_DIRECTIVE_NEW

# Minimum fraction of a claim's content words that must also appear in the report for
# the claim to be considered "actually in the report". A faithful paraphrase shares most
# of its content words with the source sentence; a fabricated strawman shares few. This
# is a deterministic proxy (no extra LLM call), not a perfect entailment check.
_IN_REPORT_THRESHOLD = 0.5

_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass
class ClaimCheck:
    """A single factual claim, whether the sources support it, and whether it actually
    appears in the report (fabrication guard)."""

    claim: str
    supported: bool
    in_report: bool = True


@dataclass
class VerificationReport:
    """The outcome of verifying a report's claims, with grounding statistics."""

    claims: list[ClaimCheck]

    @property
    def total(self) -> int:
        return len(self.claims)

    @property
    def supported(self) -> int:
        return sum(1 for check in self.claims if check.supported)

    @property
    def unsupported_claims(self) -> list[str]:
        return [check.claim for check in self.claims if not check.supported]

    @property
    def fabricated(self) -> int:
        """How many extracted claims do NOT actually appear in the report."""
        return sum(1 for check in self.claims if not check.in_report)

    @property
    def percent_grounded(self) -> float:
        return 100.0 * self.supported / self.total if self.total else 0.0


def _content_tokens(text: str) -> set[str]:
    """Lowercased content words (length > 3) used for the in-report overlap heuristic."""
    return {token for token in _WORD_RE.findall(text.lower()) if len(token) > 3}


def claim_in_report(claim: str, report_body: str) -> bool:
    """Heuristic: does `claim` actually appear in `report_body`?

    Measures the fraction of the claim's content words present in the report. Returns
    True when that fraction meets `_IN_REPORT_THRESHOLD`. If the claim has no content
    words to compare, we do not flag it (return True).
    """
    claim_tokens = _content_tokens(claim)
    if not claim_tokens:
        return True
    report_tokens = _content_tokens(report_body)
    overlap = len(claim_tokens & report_tokens) / len(claim_tokens)
    return overlap >= _IN_REPORT_THRESHOLD


def verify_report(
    client: Groq, report_body: str, sources: list[Source]
) -> VerificationReport:
    """Verify the report's claims against `sources` (one batched LLM call).

    Returns an empty report (no claims) without calling the model if there is nothing
    to verify, or if the model's reply can't be parsed — degrading gracefully.
    """
    if not report_body.strip() or not sources:
        return VerificationReport(claims=[])

    source_text = "\n\n".join(
        f"[{i}] {source.title} ({source.url})\n{source.text}"
        for i, source in enumerate(sources, start=1)
    )
    system, directive = _select_prompt()
    user = f"Sources:\n{source_text}\n\nReport:\n{report_body}\n\n{directive}"

    try:
        data = llm.complete_json(
            client,
            model=config.VERIFY_MODEL,
            system=system,
            user=user,
            max_tokens=4000,
        )
        # Tolerate either a bare array of claim objects or an object wrapping them
        # under a key (e.g. {"claims": [...]}).
        checks: list[ClaimCheck] = []
        for item in llm.extract_list(data):
            if not isinstance(item, dict):
                continue
            claim = str(item.get("claim", "")).strip()
            if claim:
                checks.append(
                    ClaimCheck(
                        claim=claim,
                        supported=bool(item.get("supported")),
                        in_report=claim_in_report(claim, report_body),
                    )
                )
    except (ValueError, AttributeError):
        return VerificationReport(claims=[])

    report = VerificationReport(claims=checks)
    if report.fabricated:
        logger.debug(
            "%d of %d extracted claim(s) do not appear in the report (possible fabrication)",
            report.fabricated, report.total,
        )
    return report


def flag_unsupported(report_body: str, verification: VerificationReport) -> str:
    """Append a clearly marked section listing any unsupported claims."""
    unsupported = verification.unsupported_claims
    if not unsupported:
        return report_body
    listed = "\n".join(f"- {claim}" for claim in unsupported)
    return (
        f"{report_body}\n\n"
        "[!] Unverified claims (not supported by the retrieved sources):\n"
        f"{listed}"
    )


def grounding_summary(verification: VerificationReport) -> str:
    """One-line grounding metric to print after the report."""
    if verification.total == 0:
        return "Grounding summary: no factual claims were identified to verify."
    return (
        f"Grounding summary: {verification.supported}/{verification.total} claims "
        f"supported ({verification.percent_grounded:.0f}% grounded)."
    )
