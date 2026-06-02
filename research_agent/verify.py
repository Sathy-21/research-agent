"""Claim verification (Phase 3): check the composed report against the source text.

Defends against a real failure: a low-quality, off-topic source (a generic library
fact-checking guide) was retrieved, trusted, and woven into a technical report. After
the report is composed, we extract its factual claims and check each one against the
ACTUAL retrieved source text — in ONE batched call, not one per claim — then report a
grounding metric.

We *flag* unsupported claims rather than deleting them. Surgically removing sentences
from finished prose would need another LLM rewrite call (cost on a rate-limited tier)
and risks mangling the text; flagging keeps the report intact and is transparent about
exactly what could not be grounded in the sources.
"""

from __future__ import annotations

from dataclasses import dataclass

from groq import Groq

from . import config, llm
from .retrieval import Source

_VERIFY_SYSTEM = (
    "You are a strict fact-checker. You are given source excerpts and a report. Break the "
    "report into its distinct factual claims. For each claim, decide whether it is "
    "supported by the source excerpts: a claim is 'supported' only if the sources "
    "directly back it up. If the sources do not mention it, mark it unsupported. Judge "
    "ONLY against the provided sources, never against prior knowledge."
)


@dataclass
class ClaimCheck:
    """A single factual claim and whether the sources support it."""

    claim: str
    supported: bool


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
    def percent_grounded(self) -> float:
        return 100.0 * self.supported / self.total if self.total else 0.0


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
    user = (
        f"Sources:\n{source_text}\n\n"
        f"Report:\n{report_body}\n\n"
        'Respond with JSON only: '
        '{"claims": [{"claim": "<claim text>", "supported": true|false}, ...]}'
    )

    try:
        data = llm.complete_json(
            client,
            model=config.VERIFY_MODEL,
            system=_VERIFY_SYSTEM,
            user=user,
            max_tokens=4000,
        )
        checks: list[ClaimCheck] = []
        for item in data.get("claims", []) or []:
            claim = str(item.get("claim", "")).strip()
            if claim:
                checks.append(ClaimCheck(claim=claim, supported=bool(item.get("supported"))))
    except (ValueError, AttributeError):
        return VerificationReport(claims=[])

    return VerificationReport(claims=checks)


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
