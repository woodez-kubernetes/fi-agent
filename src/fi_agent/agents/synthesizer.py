"""Executive summary. One LLM call, prose only - every figure around it is rendered
from data."""

from __future__ import annotations

import logging

from fi_agent.agents.prompts import SYNTHESIZER_SYSTEM, synthesizer_user
from fi_agent.llm import LLMClient
from fi_agent.schemas import ExecutiveSummary, Finding, MarketContext

log = logging.getLogger(__name__)


def _line(finding: Finding) -> str:
    quote = finding.mover.quote
    base = f"{finding.mover.symbol} ({finding.mover.name}) {quote.pct_change:+.2f}%"
    if finding.analysis:
        return f"{base}: {finding.analysis.headline} [driver: {finding.analysis.driver}]"
    return f"{base}: no analysis available"


def fallback_summary(findings: list[Finding], context: MarketContext) -> ExecutiveSummary:
    """Deterministic summary used when the model is unavailable or the run is data-only."""
    if not findings:
        return ExecutiveSummary(
            summary=(
                f"No watchlist names cleared the screening thresholds. "
                f"{context.benchmark} is {context.benchmark_pct:+.2f}% on the session."
            ),
            themes=[],
            top_story="",
        )
    ranked = sorted(findings, key=lambda f: abs(f.mover.quote.pct_change), reverse=True)
    names = ", ".join(
        f"{f.mover.symbol} {f.mover.quote.pct_change:+.2f}%" for f in ranked[:5]
    )
    return ExecutiveSummary(
        summary=(
            f"{len(findings)} watchlist name(s) cleared screening: {names}. "
            f"{context.benchmark} is {context.benchmark_pct:+.2f}% on the session. "
            "Narrative analysis was not available for this run."
        ),
        themes=[],
        top_story=ranked[0].mover.symbol,
    )


def synthesize(
    client: LLMClient, findings: list[Finding], context: MarketContext
) -> ExecutiveSummary:
    if not findings:
        return fallback_summary(findings, context)

    result = client.structured(
        ExecutiveSummary,
        SYNTHESIZER_SYSTEM,
        synthesizer_user([_line(f) for f in findings], context.benchmark, context.benchmark_pct),
        label="synthesizer",
    )
    if result is None:
        log.warning("synthesizer unavailable, using deterministic summary")
        return fallback_summary(findings, context)

    # The model sometimes returns a name rather than a symbol here.
    symbols = {f.mover.symbol for f in findings}
    if result.top_story not in symbols:
        result.top_story = max(
            findings, key=lambda f: abs(f.mover.quote.pct_change)
        ).mover.symbol
    return result
