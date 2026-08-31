"""Shared fixtures. Everything here is offline and deterministic."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fi_agent.config import (
    LLMSettings,
    NewsSettings,
    PathSettings,
    RunSettings,
    ScreeningSettings,
    Settings,
    TickerConfig,
    Watchlist,
)
from fi_agent.schemas import (
    AnalystResult,
    Article,
    EvidenceItem,
    ExecutiveSummary,
    MarketContext,
    Mover,
    Quote,
    TriageResult,
    TriageSelection,
    VerifierResult,
)

NOW = datetime(2026, 8, 30, 20, 0, tzinfo=UTC)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        llm=LLMSettings(base_url="http://test:11434", model="qwen2.5:7b"),
        screening=ScreeningSettings(),
        news=NewsSettings(lookback_hours=36),
        run=RunSettings(budget_seconds=60, max_movers=8),
        paths=PathSettings(
            reports_dir=str(tmp_path / "reports"), database=str(tmp_path / "db.sqlite")
        ),
    )


@pytest.fixture
def watchlist() -> Watchlist:
    return Watchlist(
        tickers=[
            TickerConfig(symbol="NVDA", name="NVIDIA", sector_etf="SMH", move_threshold_pct=4.5),
            TickerConfig(symbol="AAPL", name="Apple", sector_etf="XLK"),
            TickerConfig(symbol="GOOGL", name="Alphabet", sector_etf="XLC", aliases=["Google"]),
        ]
    )


def make_quote(symbol: str, price: float, prev_close: float, **kwargs) -> Quote:
    defaults = {
        "volume": 1_000_000.0,
        "avg_volume_30d": 1_000_000.0,
        "week52_high": max(price, prev_close) * 1.5,
        "week52_low": min(price, prev_close) * 0.5,
        "as_of": NOW,
        "history": [prev_close] * 29 + [price],
    }
    defaults.update(kwargs)
    return Quote(symbol=symbol, price=price, prev_close=prev_close, **defaults)


@pytest.fixture
def quote_factory():
    return make_quote


@pytest.fixture
def context() -> MarketContext:
    return MarketContext(
        benchmark="SPY",
        benchmark_pct=-0.25,
        sector_pct={"SMH": -3.5, "XLK": -1.0, "XLC": 1.4},
        as_of=NOW,
    )


@pytest.fixture
def mover(context) -> Mover:
    return Mover(
        symbol="NVDA",
        name="NVIDIA",
        sector_etf="SMH",
        quote=make_quote("NVDA", 217.55, 227.98, volume=1_500_000.0),
        beta=2.07,
        residual_pct=-4.05,
        sector_pct=-3.5,
        triggers=["moved -4.57% (threshold 4.5%)"],
    )


@pytest.fixture
def articles() -> list[Article]:
    return [
        Article(
            url="https://example.com/a",
            title="SK Hynix warns memory shortage will persist",
            source="Reuters",
            published_at=NOW - timedelta(hours=3),
            text="Chip stocks slid after SK Hynix warned that memory supply will stay tight.",
        ),
        Article(
            url="https://example.com/b",
            title="Nvidia falls with chip peers",
            source="Barrons",
            published_at=NOW - timedelta(hours=5),
            summary="Nvidia shares fell alongside peers with no company announcement.",
        ),
    ]


class FakeLLM:
    """Stand-in for LLMClient returning canned, schema-valid objects.

    Lets the whole graph be exercised offline in milliseconds instead of minutes.
    """

    def __init__(self, **overrides) -> None:
        self.calls = 0
        self.total_seconds = 0.0
        self.labels: list[str] = []
        self.overrides = overrides

    def check(self):
        return True, "fake"

    def structured(self, schema, system, user, label=""):
        self.calls += 1
        self.labels.append(label)
        if schema.__name__ in self.overrides:
            return self.overrides[schema.__name__]
        return {
            "TriageResult": TriageResult(
                selected=[TriageSelection(idx=0, relevance="high", why="direct cause")]
            ),
            "AnalystResult": AnalystResult(
                headline="Nvidia slides with chip sector on memory warning",
                narrative="Chip names fell together after a supply warning.",
                driver="sector",
                confidence="medium",
                evidence=[EvidenceItem(claim="SK Hynix warned on memory supply", source_idx=0)],
                watch_next="Whether peers recover on Monday.",
            ),
            "VerifierResult": VerifierResult(drops=[]),
            "ExecutiveSummary": ExecutiveSummary(
                summary="Chip weakness drove the session.",
                themes=["semiconductor weakness"],
                top_story="NVDA",
            ),
        }.get(schema.__name__)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()
