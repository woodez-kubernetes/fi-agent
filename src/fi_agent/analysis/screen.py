"""Mover screening - the gate that decides which tickers are worth LLM attention.

This is the single most important cost control in the system. The Ollama server
serialises inference, so every ticker that reaches the agents adds real wall-clock
time. Screening keeps that list short and, more importantly, keeps it *interesting*.

The highest-signal rule is the beta-adjusted residual. A raw percentage threshold
answers "did this move a lot?", which on a day the whole market fell 3% flags
everything and explains nothing. The residual answers "did this move for reasons of
its own?", which is the question the report exists to answer.
"""

from __future__ import annotations

import logging

import pandas as pd

from fi_agent.config import ScreeningSettings, Watchlist
from fi_agent.data.market import compute_beta, market_is_open, session_fraction
from fi_agent.schemas import MarketContext, Mover, Quote

log = logging.getLogger(__name__)


def build_market_context(
    quotes: dict[str, Quote], watchlist: Watchlist, settings: ScreeningSettings
) -> MarketContext:
    """Benchmark and sector baselines for the session."""
    benchmark_quote = quotes.get(settings.benchmark)
    sector_pct: dict[str, float] = {}
    for etf in {t.sector_etf for t in watchlist.tickers if t.sector_etf}:
        quote = quotes.get(etf)
        if quote:
            sector_pct[etf] = round(quote.pct_change, 3)

    return MarketContext(
        benchmark=settings.benchmark,
        benchmark_pct=round(benchmark_quote.pct_change, 3) if benchmark_quote else 0.0,
        sector_pct=sector_pct,
        as_of=benchmark_quote.as_of if benchmark_quote else None,
    )


def residual_pct(quote: Quote, beta: float | None, benchmark_pct: float) -> float | None:
    """Move not explained by the benchmark: actual - beta * benchmark."""
    if beta is None:
        return None
    return round(quote.pct_change - beta * benchmark_pct, 3)


def effective_volume_ratio(quote: Quote) -> float | None:
    """Volume ratio, prorated for how much of the session has elapsed.

    Comparing 10:00am volume against a full-day average would make every intraday run
    look quiet, so the average is scaled to the fraction of the session so far.
    """
    if not quote.volume or not quote.avg_volume_30d:
        return None
    expected = quote.avg_volume_30d * (session_fraction() if market_is_open() else 1.0)
    if expected <= 0:
        return None
    return round(quote.volume / expected, 3)


def screen(
    quotes: dict[str, Quote],
    watchlist: Watchlist,
    context: MarketContext,
    settings: ScreeningSettings,
    frame: pd.DataFrame | None = None,
    max_movers: int = 8,
) -> tuple[list[Mover], list[Mover]]:
    """Split the watchlist into (movers, quiet).

    Returns Movers for every configured ticker; the first list cleared at least one
    trigger, the second did not. Both are needed - the report shows quiet names too.
    """
    movers: list[Mover] = []
    quiet: list[Mover] = []

    for ticker in watchlist.tickers:
        quote = quotes.get(ticker.symbol)
        if quote is None:
            log.warning("no quote for %s, skipping", ticker.symbol)
            continue

        beta = (
            compute_beta(frame, ticker.symbol, settings.benchmark, settings.beta_lookback_days)
            if frame is not None
            else None
        )
        residual = residual_pct(quote, beta, context.benchmark_pct)
        vol_ratio = effective_volume_ratio(quote)
        threshold = ticker.move_threshold_pct or settings.move_threshold_pct

        triggers: list[str] = []
        if abs(quote.pct_change) >= threshold:
            triggers.append(f"moved {quote.pct_change:+.2f}% (threshold {threshold:.1f}%)")
        if vol_ratio is not None and vol_ratio >= settings.volume_multiple:
            triggers.append(f"volume {vol_ratio:.1f}x average")
        gap = quote.gap_pct
        if gap is not None and abs(gap) >= settings.gap_pct:
            triggers.append(f"gapped {gap:+.2f}% at the open")
        if residual is not None and abs(residual) >= settings.idio_threshold_pct:
            triggers.append(
                f"{residual:+.2f}% move unexplained by {settings.benchmark} (beta {beta:.2f})"
            )
        if settings.flag_52w_extremes:
            if quote.at_52w_high:
                triggers.append("at a 52-week high")
            elif quote.at_52w_low:
                triggers.append("at a 52-week low")

        mover = Mover(
            symbol=ticker.symbol,
            name=ticker.display_name(),
            sector_etf=ticker.sector_etf,
            quote=quote,
            beta=beta,
            residual_pct=residual,
            sector_pct=context.sector_pct.get(ticker.sector_etf or ""),
            triggers=triggers,
        )
        (movers if triggers else quiet).append(mover)

    # Largest absolute idiosyncratic move first; fall back to raw move when beta is
    # unavailable, so a ticker is never ranked above one with a real residual.
    movers.sort(
        key=lambda m: abs(m.residual_pct if m.residual_pct is not None else m.quote.pct_change),
        reverse=True,
    )

    if len(movers) > max_movers:
        log.info("capping %d movers at %d", len(movers), max_movers)
        quiet.extend(movers[max_movers:])
        movers = movers[:max_movers]

    return movers, quiet
