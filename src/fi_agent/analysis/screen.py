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
from datetime import datetime

import pandas as pd

from fi_agent.config import ScreeningSettings, Watchlist
from fi_agent.data.market import (
    compute_beta,
    intraday_volume_fraction,
    market_is_open,
    volume_estimate_is_reliable,
)
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


def effective_volume_ratio(
    quote: Quote, curve: float = 0.7, now: datetime | None = None
) -> float | None:
    """Volume so far against what would be normal by this point in the session.

    The expected share comes from a U-shaped intraday curve rather than from clock time,
    because volume clusters at the open and close. Scaling by clock time instead made
    ordinary mornings look like volume spikes - at 4.9% into a session it put 8 of 10
    watchlist names above their 30-day average and tripped the trigger on two of them.
    """
    if not quote.volume or not quote.avg_volume_30d:
        return None
    expected_share = intraday_volume_fraction(now, curve) if market_is_open(now) else 1.0
    expected = quote.avg_volume_30d * expected_share
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
    now: datetime | None = None,
) -> tuple[list[Mover], list[Mover]]:
    """Split the watchlist into (movers, quiet).

    Returns Movers for every configured ticker; the first list cleared at least one
    trigger, the second did not. Both are needed - the report shows quiet names too.
    """
    movers: list[Mover] = []
    quiet: list[Mover] = []

    # Volume is still shown during the opening minutes, but it must not flag anything:
    # the opening auction prints in a burst no smooth curve models well.
    volume_trusted = volume_estimate_is_reliable(now, settings.volume_warmup_minutes)
    if not volume_trusted:
        log.info("within the opening warmup, volume triggers suppressed")

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
        vol_ratio = effective_volume_ratio(quote, settings.intraday_volume_curve, now)
        threshold = ticker.move_threshold_pct or settings.move_threshold_pct

        triggers: list[str] = []
        if abs(quote.pct_change) >= threshold:
            triggers.append(f"moved {quote.pct_change:+.2f}% (threshold {threshold:.1f}%)")
        if volume_trusted and vol_ratio is not None and vol_ratio >= settings.volume_multiple:
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
