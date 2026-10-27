"""Market data via yfinance.

Everything comes from a single batched daily-bar download rather than yfinance's
`.info` endpoint, which is slow and the most breakage-prone part of the library.
Price, previous close, gap, volume ratio, 52-week extremes and beta are all derived
from that one frame.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

from fi_agent.schemas import Quote

log = logging.getLogger(__name__)

MARKET_TZ = ZoneInfo("America/New_York")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
HISTORY_POINTS = 30


def market_is_open(now: datetime | None = None) -> bool:
    """Approximate NYSE session check. Weekday and clock only - holidays are not modelled,
    so a holiday looks like a session with no volume, which screening ignores anyway."""
    current = (now or datetime.now(UTC)).astimezone(MARKET_TZ)
    if current.weekday() >= 5:
        return False
    return MARKET_OPEN <= current.time() < MARKET_CLOSE


def session_fraction(now: datetime | None = None) -> float:
    """Fraction of the trading session elapsed by the clock, 0.0-1.0."""
    current = (now or datetime.now(UTC)).astimezone(MARKET_TZ)
    if not market_is_open(current):
        return 1.0
    elapsed = (current - _session_open(current)).total_seconds()
    total = (_session_close(current) - _session_open(current)).total_seconds()
    return max(0.0, min(1.0, elapsed / total))


def _session_open(current: datetime) -> datetime:
    return current.replace(
        hour=MARKET_OPEN.hour, minute=MARKET_OPEN.minute, second=0, microsecond=0
    )


def _session_close(current: datetime) -> datetime:
    return current.replace(
        hour=MARKET_CLOSE.hour, minute=MARKET_CLOSE.minute, second=0, microsecond=0
    )


def minutes_since_open(now: datetime | None = None) -> float:
    """Minutes elapsed since the opening bell. Negative before the open."""
    current = (now or datetime.now(UTC)).astimezone(MARKET_TZ)
    return (current - _session_open(current)).total_seconds() / 60.0


def intraday_volume_fraction(now: datetime | None = None, curve: float = 0.7) -> float:
    """Expected fraction of a day's volume traded by now, 0.0-1.0.

    Intraday volume is U-shaped, not uniform: the opening auction and the closing hour
    carry far more than their share of the day, and the middle of the session is quiet.
    Prorating the 30-day average by clock time alone therefore understates what "normal"
    volume looks like in the morning, which in turn inflates every volume ratio and
    flags half the watchlist just after the open.

    Cumulative volume is modelled as

        cum(t) = t + (c / 2pi) * sin(2 * pi * t)

    whose density, 1 + c * cos(2 * pi * t), peaks at the open and close and troughs at
    midday. `curve` is c: 0 reduces this to the old linear behaviour, 1 is the most
    pronounced U that keeps the density non-negative.
    """
    fraction = session_fraction(now)
    if fraction >= 1.0:
        return 1.0
    c = max(0.0, min(1.0, curve))
    adjusted = fraction + (c / (2 * math.pi)) * math.sin(2 * math.pi * fraction)
    # Floor it so the very first seconds of the session cannot divide by ~zero.
    return max(0.005, min(1.0, adjusted))


def volume_estimate_is_reliable(
    now: datetime | None = None, warmup_minutes: float = 15.0
) -> bool:
    """Whether enough of the session has passed to judge volume.

    The opening auction prints in a burst that no smooth curve models well, so for the
    first few minutes any ratio is noise dressed as signal.
    """
    if not market_is_open(now):
        return True
    return minutes_since_open(now) >= warmup_minutes


def _column(frame: pd.DataFrame, field: str, symbol: str) -> pd.Series | None:
    """Pull one (field, symbol) series out of yfinance's MultiIndex frame.

    yfinance returns MultiIndex columns for multiple symbols but flat columns for one,
    so both shapes have to be handled.
    """
    try:
        if isinstance(frame.columns, pd.MultiIndex):
            return frame[(field, symbol)].dropna()
        return frame[field].dropna()
    except (KeyError, IndexError):
        return None


def download_history(symbols: list[str], period: str = "1y") -> pd.DataFrame:
    """One batched download for every symbol the run needs."""
    unique = sorted(set(symbols))
    if not unique:
        return pd.DataFrame()
    log.debug("downloading %d symbols, period=%s", len(unique), period)
    frame = yf.download(
        unique,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    if frame is None or frame.empty:
        raise RuntimeError(f"yfinance returned no data for {unique}")
    return frame


def quote_from_history(symbol: str, frame: pd.DataFrame) -> Quote | None:
    """Build a Quote from daily bars. Returns None when the symbol has no usable data."""
    closes = _column(frame, "Close", symbol)
    if closes is None or len(closes) < 2:
        log.warning("no close history for %s", symbol)
        return None

    opens = _column(frame, "Open", symbol)
    highs = _column(frame, "High", symbol)
    lows = _column(frame, "Low", symbol)
    volumes = _column(frame, "Volume", symbol)

    price = float(closes.iloc[-1])
    prev_close = float(closes.iloc[-2])

    volume = float(volumes.iloc[-1]) if volumes is not None and len(volumes) else None
    # Average over the prior 30 sessions, excluding today's partial bar.
    avg_volume = None
    if volumes is not None and len(volumes) > 2:
        prior = volumes.iloc[-31:-1]
        if len(prior):
            avg_volume = float(prior.mean())

    window = closes.iloc[-252:]
    return Quote(
        symbol=symbol,
        price=price,
        prev_close=prev_close,
        open=float(opens.iloc[-1]) if opens is not None and len(opens) else None,
        day_high=float(highs.iloc[-1]) if highs is not None and len(highs) else None,
        day_low=float(lows.iloc[-1]) if lows is not None and len(lows) else None,
        volume=volume,
        avg_volume_30d=avg_volume,
        week52_high=float(window.max()) if len(window) else None,
        week52_low=float(window.min()) if len(window) else None,
        as_of=datetime.now(UTC),
        history=[float(v) for v in closes.iloc[-HISTORY_POINTS:]],
    )


def fetch_quotes(symbols: list[str]) -> tuple[dict[str, Quote], pd.DataFrame, list[str]]:
    """Fetch quotes for every symbol. Returns (quotes, raw frame, failed symbols)."""
    frame = download_history(symbols)
    quotes: dict[str, Quote] = {}
    failed: list[str] = []
    for symbol in symbols:
        quote = quote_from_history(symbol, frame)
        if quote is None:
            failed.append(symbol)
        else:
            quotes[symbol] = quote
    return quotes, frame, failed


def compute_beta(
    frame: pd.DataFrame, symbol: str, benchmark: str, lookback_days: int = 60
) -> float | None:
    """Ordinary-least-squares beta of daily returns against the benchmark.

    Returns None when there is not enough overlapping history to be meaningful.
    """
    if symbol == benchmark:
        return 1.0
    stock = _column(frame, "Close", symbol)
    bench = _column(frame, "Close", benchmark)
    if stock is None or bench is None:
        return None

    joined = pd.concat([stock, bench], axis=1, keys=["stock", "bench"]).dropna()
    returns = joined.pct_change().dropna().iloc[-lookback_days:]
    if len(returns) < 20:
        return None

    x = returns["bench"].to_numpy(dtype=float)
    y = returns["stock"].to_numpy(dtype=float)
    variance = float(np.var(x))
    if variance == 0.0:
        return None
    beta = float(np.cov(y, x, ddof=0)[0][1] / variance)
    # Guard against degenerate fits from thin or stale data.
    if not np.isfinite(beta) or abs(beta) > 5:
        return None
    return round(beta, 3)
