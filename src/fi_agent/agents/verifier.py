"""Verification pass: strike claims their cited article does not support.

This is the system's main defence against a small model inventing causality. It runs as
a single batched LLM call across every mover, so its cost does not grow with the size of
the watchlist.

A deterministic pre-pass runs first and needs no model at all.
"""

from __future__ import annotations

import logging

from fi_agent.agents.prompts import VERIFIER_SYSTEM, verifier_user
from fi_agent.llm import LLMClient
from fi_agent.schemas import Finding, VerifierResult, downgrade

log = logging.getLogger(__name__)

EXCERPT_CHARS = 1200


def verify(client: LLMClient, findings: list[Finding]) -> list[Finding]:
    """Remove unsupported evidence in place and return the findings."""
    entries: list[tuple[str, int, str, str]] = []
    for finding in findings:
        if finding.degraded or finding.analysis is None:
            continue
        for claim_idx, item in enumerate(finding.analysis.evidence):
            article = finding.articles[item.source_idx]
            excerpt = (article.body or article.title)[:EXCERPT_CHARS]
            entries.append((finding.mover.symbol, claim_idx, item.claim, excerpt))

    if not entries:
        return findings

    result = client.structured(
        VerifierResult, VERIFIER_SYSTEM, verifier_user(entries), label="verifier"
    )
    if result is None:
        # Verification is a safety net, not a gate. If it fails, the report ships with
        # its claims intact and the diagnostics note that the pass did not run.
        log.warning("verification pass unavailable; claims retained unverified")
        return findings

    drops: dict[str, set[int]] = {}
    for drop in result.drops:
        drops.setdefault(drop.symbol, set()).add(drop.claim_idx)

    for finding in findings:
        if finding.analysis is None:
            continue
        to_drop = drops.get(finding.mover.symbol, set())
        if not to_drop:
            continue
        kept = []
        for claim_idx, item in enumerate(finding.analysis.evidence):
            if claim_idx in to_drop:
                finding.dropped_claims.append(item.claim)
            else:
                kept.append(item)
        if len(kept) != len(finding.analysis.evidence):
            log.info(
                "%s: dropped %d unsupported claim(s)",
                finding.mover.symbol,
                len(finding.analysis.evidence) - len(kept),
            )
            finding.analysis.evidence = kept
            finding.analysis.confidence = downgrade(finding.analysis.confidence)

    return findings
