"""Screening is the gate that decides what the LLM ever sees, so its arithmetic is
tested directly rather than through the graph."""

from __future__ import annotations

from fi_agent.analysis.screen import build_market_context, residual_pct, screen
from fi_agent.config import ScreeningSettings, TickerConfig, Watchlist
from fi_agent.schemas import MarketContext

from .conftest import make_quote


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

    movers, quiet = screen(quotes, watchlist, context, settings)
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
    movers, quiet = screen(quotes, watchlist, context, settings)
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
    movers, _ = screen(quotes, watchlist, context, settings)
    assert movers and "volume" in movers[0].triggers[0]


def test_per_ticker_threshold_override():
    settings = ScreeningSettings(move_threshold_pct=3.0, idio_threshold_pct=99.0)
    watchlist = Watchlist(
        tickers=[TickerConfig(symbol="HIGHBETA", name="Volatile", move_threshold_pct=6.0)]
    )
    context = MarketContext(benchmark="SPY", benchmark_pct=0.0)
    quotes = {"HIGHBETA": make_quote("HIGHBETA", price=104.0, prev_close=100.0)}

    movers, quiet = screen(quotes, watchlist, context, settings)
    assert [m.symbol for m in quiet] == ["HIGHBETA"], "4% must not clear a 6% override"


def test_52_week_high_is_flagged():
    settings = ScreeningSettings(flag_52w_extremes=True, idio_threshold_pct=99.0)
    watchlist = Watchlist(tickers=[TickerConfig(symbol="X", name="X Corp")])
    context = MarketContext(benchmark="SPY", benchmark_pct=0.0)
    quotes = {
        "X": make_quote("X", price=200.0, prev_close=199.5, week52_high=200.0, week52_low=100.0)
    }
    movers, _ = screen(quotes, watchlist, context, settings)
    assert movers and any("52-week high" in t for t in movers[0].triggers)


def test_max_movers_caps_llm_work():
    settings = ScreeningSettings(move_threshold_pct=0.1)
    watchlist = Watchlist(
        tickers=[TickerConfig(symbol=f"T{i}", name=f"T{i}") for i in range(10)]
    )
    context = MarketContext(benchmark="SPY", benchmark_pct=0.0)
    quotes = {f"T{i}": make_quote(f"T{i}", price=100.0 + i, prev_close=100.0) for i in range(10)}

    movers, quiet = screen(quotes, watchlist, context, settings, max_movers=3)
    assert len(movers) == 3
    assert len(quiet) == 7, "capped movers must still appear in the report"


def test_missing_quote_is_skipped_not_fatal():
    settings = ScreeningSettings()
    watchlist = Watchlist(
        tickers=[TickerConfig(symbol="GOOD", name="G"), TickerConfig(symbol="GONE", name="X")]
    )
    context = MarketContext(benchmark="SPY", benchmark_pct=0.0)
    quotes = {"GOOD": make_quote("GOOD", price=100.0, prev_close=100.0)}

    movers, quiet = screen(quotes, watchlist, context, settings)
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
