"""Turn a finished run state into HTML and Markdown.

The template receives a flat view model rather than raw graph state, so the templates
carry no logic beyond formatting. Every number in the output comes from `Quote` and
`Mover` fields computed in Python.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from fi_agent.data.market import market_is_open
from fi_agent.report.sparkline import sparkline
from fi_agent.schemas import ExecutiveSummary, Finding, MarketContext, Mover

TEMPLATE_DIR = Path(__file__).parent / "templates"


@dataclass
class Diagnostics:
    model: str
    base_url: str
    llm_calls: int
    llm_seconds: float
    duration_s: float
    degraded: list[str]


def _environment() -> Environment:
    # autoescape is forced on rather than delegated to select_autoescape, which keys off
    # the filename extension and would see ".j2" and disable escaping. Article titles
    # come from arbitrary RSS feeds and narratives come from a language model, so
    # everything interpolated here is untrusted; only the sparkline is marked |safe.
    return Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )


class _FindingView:
    """Adapter exposing the few derived values the templates need."""

    def __init__(self, finding: Finding) -> None:
        self._finding = finding
        self.mover = finding.mover
        self.articles = finding.articles
        self.analysis = finding.analysis
        self.dropped_claims = finding.dropped_claims
        self.degraded = finding.degraded
        self.degraded_reason = finding.degraded_reason
        self.cited = finding.cited_articles
        self.sparkline = sparkline(
            finding.mover.quote.history,
            up=finding.mover.quote.pct_change >= 0,
            label=f"{finding.mover.symbol} last {len(finding.mover.quote.history)} sessions",
        )


def build_view(
    run_id: str,
    findings: list[Finding],
    quiet: list[Mover],
    context: MarketContext,
    summary: ExecutiveSummary,
    errors: list[str],
    diagnostics: Diagnostics,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    # Fan-in order is nondeterministic, so restore the screening rank: biggest
    # idiosyncratic move first.
    ordered = sorted(
        findings,
        key=lambda f: abs(
            f.mover.residual_pct
            if f.mover.residual_pct is not None
            else f.mover.quote.pct_change
        ),
        reverse=True,
    )
    return {
        "run_id": run_id,
        "generated_at": generated_at or datetime.now(UTC),
        "market_open": market_is_open(),
        "findings": [_FindingView(f) for f in ordered],
        "quiet": sorted(quiet, key=lambda m: abs(m.quote.pct_change), reverse=True),
        "context": context,
        "summary": summary,
        "errors": errors,
        "diagnostics": diagnostics,
    }


def render_html(view: dict[str, Any]) -> str:
    return _environment().get_template("report.html.j2").render(**view)


def render_markdown(view: dict[str, Any]) -> str:
    # Markdown must not be HTML-escaped, so it renders through a separate environment.
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR), trim_blocks=True, lstrip_blocks=True
    )
    return env.get_template("report.md.j2").render(**view)
