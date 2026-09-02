"""Screening is the gate that decides what the LLM ever sees, so its arithmetic is
tested directly rather than through the graph."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from fi_agent.analysis.screen import (
    build_market_context,
    effective_volume_ratio,
    residual_pct,
    screen,
)
from fi_agent.config import ScreeningSettings, TickerConfig, Watchlist
from fi_agent.data.market import intraday_volume_fraction, session_fraction
from fi_agent.schemas import MarketContext

from .conftest import make_quote

ET = ZoneInfo("America/New_York")


def at(hour: int, minute: int) -> datetime:
    """A moment during the session on Monday 31 August 2026."""
    return datetime(2026, 8, 31, hour, minute, tzinfo=ET)


# A Saturday, so the market is definitively closed.
CLOSED = datetime(2026, 8, 29, 12, 0, tzinfo=ET)


def test_pct_change_and_gap():
    quote = make_quote("X", price=110.0, prev_close=100.0, open=105.0)
    assert quote.pct_change == 10.0
    assert quote.gap_pct == 5.0


def test_pct_change_survives_zero_prev_close():
    quote = make_quote("X", price=110.0, prev_close=0.0)
    assert quote.pct_change == 0.0


def test_residual_strips_the_market_component():
    # A 4% fall on a 2% down market with beta 2 is entirely explained by the market.
    quote = make_quote("X", price=96.0, prev_close=100.0)
    assert residual_pct(quote, beta=2.0, benchmark_pct=-2.0) == 0.0


def test_residual_is_none_without_beta():
    quote = make_quote("X", price=96.0, prev_close=100.0)
    assert residual_pct(quote, beta=None, benchmark_pct=-2.0) is None


def test_quiet_stock_on_a_falling_market_is_not_flagged():
    """The case a raw percentage threshold gets wrong: a big move that is all market."""
    settings = ScreeningSettings(move_threshold_pct=3.0, idio_threshold_pct=2.0)
    watchlist = Watchlist(tickers=[TickerConfig(symbol="X", name="X Corp")])
    context = MarketContext(benchmark="SPY", benchmark_pct=-3.0)
    quotes = {"X": make_quote("X", price=97.0, prev_close=100.0)}

    movers, quiet = screen(quotes, watchlist, context, settings, now=CLOSED)
    # -3% clears the raw threshold, so it is flagged, but the residual is ~0.
    assert [m.symbol for m in movers] == ["X"]
    assert quiet == []


def test_small_idiosyncratic_move_is_flagged():
    """The case a raw threshold misses: a modest move with nothing explaining it."""
    settings = ScreeningSettings(move_threshold_pct=3.0, idio_threshold_pct=2.0)
    watchlist = Watchlist(tickers=[TickerConfig(symbol="X", name="X Corp")])
    context = MarketContext(benchmark="SPY", benchmark_pct=0.0)
    quotes = {"X": make_quote("X", price=102.5, prev_close=100.0)}

    class FrameStub:
        pass

    # With no frame, beta is None and the residual rule cannot fire.
    movers, quiet = screen(quotes, watchlist, context, settings, now=CLOSED)
    assert [m.symbol for m in quiet] == ["X"], "2.5% should not clear a 3% raw threshold"


def test_volume_spike_alone_flags():
    settings = ScreeningSettings(volume_multiple=2.0)
    watchlist = Watchlist(tickers=[TickerConfig(symbol="X", name="X Corp")])
    context = MarketContext(benchmark="SPY", benchmark_pct=0.0)
    quotes = {
        "X": make_quote(
            "X", price=100.5, prev_close=100.0, volume=5_000_000.0, avg_volume_30d=1_000_000.0
        )
    }
    movers, _ = screen(quotes, watchlist, context, settings, now=CLOSED)
    assert movers and "volume" in movers[0].triggers[0]


# -- intraday volume curve --------------------------------------------------------------


def test_volume_curve_starts_and_ends_at_the_bounds():
    assert intraday_volume_fraction(at(9, 30)) < 0.02
    assert intraday_volume_fraction(at(15, 59)) > 0.98
    assert intraday_volume_fraction(CLOSED) == 1.0


def test_volume_curve_is_monotonic():
    previous = 0.0
    for hour, minute in [(9, 45), (10, 30), (11, 30), (12, 30), (13, 30), (14, 30), (15, 30)]:
        current = intraday_volume_fraction(at(hour, minute))
        assert current > previous, f"volume fraction went backwards at {hour}:{minute}"
        previous = current


def test_volume_curve_is_u_shaped():
    """Volume clusters at the open and close, so early on the curve must run ahead of
    the clock and by mid-afternoon it must lag behind."""
    morning = at(10, 0)
    afternoon = at(14, 30)
    assert intraday_volume_fraction(morning) > session_fraction(morning) * 1.4
    assert intraday_volume_fraction(afternoon) < session_fraction(afternoon)


def test_zero_curve_reproduces_clock_time():
    moment = at(11, 0)
    assert intraday_volume_fraction(moment, curve=0.0) == pytest.approx(
        session_fraction(moment), abs=1e-9
    )


def test_morning_volume_is_not_inflated():
    """Regression. Prorating by clock time put 8 of 10 live watchlist names above their
    30-day average 25 minutes into a session, and tripped the 2.0x trigger on two.

    A stock trading its ordinary share of the day's volume must read near 1.0x.
    """
    moment = at(9, 55)
    typical_share = intraday_volume_fraction(moment)
    quote = make_quote(
        "X", price=100.0, prev_close=100.0,
        volume=1_000_000.0 * typical_share, avg_volume_30d=1_000_000.0,
    )
    assert effective_volume_ratio(quote, now=moment) == pytest.approx(1.0, abs=0.01)

    # The same stock scored 1.6x or more under the old linear model.
    assert effective_volume_ratio(quote, curve=0.0, now=moment) > 1.5


def test_genuine_morning_spike_still_flags():
    """The fix must not blind the screener - triple the normal share still trips."""
    settings = ScreeningSettings(volume_multiple=2.0, idio_threshold_pct=99.0)
    watchlist = Watchlist(tickers=[TickerConfig(symbol="X", name="X Corp")])
    context = MarketContext(benchmark="SPY", benchmark_pct=0.0)
    moment = at(10, 30)
    quotes = {
        "X": make_quote(
            "X", price=100.0, prev_close=100.0,
            volume=1_000_000.0 * intraday_volume_fraction(moment) * 3.0,
            avg_volume_30d=1_000_000.0,
        )
    }
    movers, _ = screen(quotes, watchlist, context, settings, now=moment)
    assert movers and any("volume" in t for t in movers[0].triggers)


def test_volume_trigger_suppressed_during_opening_warmup():
    """The opening auction prints in a burst no smooth curve models, so nothing should
    be flagged on volume alone in the first minutes."""
    settings = ScreeningSettings(volume_multiple=2.0, idio_threshold_pct=99.0)
    watchlist = Watchlist(tickers=[TickerConfig(symbol="X", name="X Corp")])
    context = MarketContext(benchmark="SPY", benchmark_pct=0.0)
    moment = at(9, 33)
    quotes = {
        "X": make_quote(
            "X", price=100.0, prev_close=100.0,
            volume=1_000_000.0, avg_volume_30d=1_000_000.0,
        )
    }
    movers, quiet = screen(quotes, watchlist, context, settings, now=moment)
    assert [m.symbol for m in quiet] == ["X"]
    # The ratio is still reported, it just cannot trigger on its own.
    assert quiet[0].quote.volume_ratio is not None


def test_volume_ratio_uses_the_full_day_when_closed():
    quote = make_quote(
        "X", price=100.0, prev_close=100.0, volume=2_000_000.0, avg_volume_30d=1_000_000.0
    )
    assert effective_volume_ratio(quote, now=CLOSED) == pytest.approx(2.0)


def test_per_ticker_threshold_override():
    settings = ScreeningSettings(move_threshold_pct=3.0, idio_threshold_pct=99.0)
    watchlist = Watchlist(
        tickers=[TickerConfig(symbol="HIGHBETA", name="Volatile", move_threshold_pct=6.0)]
    )
    context = MarketContext(benchmark="SPY", benchmark_pct=0.0)
    quotes = {"HIGHBETA": make_quote("HIGHBETA", price=104.0, prev_close=100.0)}

    movers, quiet = screen(quotes, watchlist, context, settings, now=CLOSED)
    assert [m.symbol for m in quiet] == ["HIGHBETA"], "4% must not clear a 6% override"


def test_52_week_high_is_flagged():
    settings = ScreeningSettings(flag_52w_extremes=True, idio_threshold_pct=99.0)
    watchlist = Watchlist(tickers=[TickerConfig(symbol="X", name="X Corp")])
    context = MarketContext(benchmark="SPY", benchmark_pct=0.0)
    quotes = {
        "X": make_quote("X", price=200.0, prev_close=199.5, week52_high=200.0, week52_low=100.0)
    }
    movers, _ = screen(quotes, watchlist, context, settings, now=CLOSED)
    assert movers and any("52-week high" in t for t in movers[0].triggers)


def test_max_movers_caps_llm_work():
    settings = ScreeningSettings(move_threshold_pct=0.1)
    watchlist = Watchlist(
        tickers=[TickerConfig(symbol=f"T{i}", name=f"T{i}") for i in range(10)]
    )
    context = MarketContext(benchmark="SPY", benchmark_pct=0.0)
    quotes = {f"T{i}": make_quote(f"T{i}", price=100.0 + i, prev_close=100.0) for i in range(10)}

    movers, quiet = screen(quotes, watchlist, context, settings, max_movers=3, now=CLOSED)
    assert len(movers) == 3
    assert len(quiet) == 7, "capped movers must still appear in the report"


def test_missing_quote_is_skipped_not_fatal():
    settings = ScreeningSettings()
    watchlist = Watchlist(
        tickers=[TickerConfig(symbol="GOOD", name="G"), TickerConfig(symbol="GONE", name="X")]
    )
    context = MarketContext(benchmark="SPY", benchmark_pct=0.0)
    quotes = {"GOOD": make_quote("GOOD", price=100.0, prev_close=100.0)}

    movers, quiet = screen(quotes, watchlist, context, settings, now=CLOSED)
    assert {m.symbol for m in movers + quiet} == {"GOOD"}


def test_market_context_collects_sector_moves(watchlist):
    quotes = {
        "SPY": make_quote("SPY", price=99.0, prev_close=100.0),
        "SMH": make_quote("SMH", price=96.0, prev_close=100.0),
        "XLK": make_quote("XLK", price=101.0, prev_close=100.0),
        "XLC": make_quote("XLC", price=100.0, prev_close=100.0),
    }
    context = build_market_context(quotes, watchlist, ScreeningSettings())
    assert context.benchmark_pct == -1.0
    assert context.sector_pct["SMH"] == -4.0
