"""Domain models and LLM output contracts.

Two rules govern everything in this module:

1. Numbers live in the *domain* models, which are populated by Python from market data.
   No LLM output model carries a price, percentage or ratio.
2. LLM output models use string enums rather than floats. A 4-bit 7B model asked for a
   0.0-1.0 confidence returns 100; asked to choose between "high"/"medium"/"low" it
   complies. Every graded field is a Literal.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Confidence = Literal["high", "medium", "low"]
Relevance = Literal["high", "medium", "low"]
Driver = Literal[
    "company_news",
    "earnings",
    "analyst_action",
    "sector",
    "macro",
    "no_identified_catalyst",
]

CONFIDENCE_ORDER: list[Confidence] = ["low", "medium", "high"]


def downgrade(confidence: Confidence) -> Confidence:
    """Step a confidence down one level, floored at 'low'."""
    idx = CONFIDENCE_ORDER.index(confidence)
    return CONFIDENCE_ORDER[max(0, idx - 1)]


# --------------------------------------------------------------------------------------
# Domain models - all numbers computed in Python
# --------------------------------------------------------------------------------------


class Quote(BaseModel):
    symbol: str
    price: float
    prev_close: float
    open: float | None = None
    day_high: float | None = None
    day_low: float | None = None
    volume: float | None = None
    avg_volume_30d: float | None = None
    week52_high: float | None = None
    week52_low: float | None = None
    currency: str = "USD"
    as_of: datetime | None = None
    history: list[float] = Field(default_factory=list, description="Trailing daily closes")

    @property
    def pct_change(self) -> float:
        if not self.prev_close:
            return 0.0
        return (self.price - self.prev_close) / self.prev_close * 100.0

    @property
    def gap_pct(self) -> float | None:
        if self.open is None or not self.prev_close:
            return None
        return (self.open - self.prev_close) / self.prev_close * 100.0

    @property
    def volume_ratio(self) -> float | None:
        if not self.volume or not self.avg_volume_30d:
            return None
        return self.volume / self.avg_volume_30d

    @property
    def at_52w_high(self) -> bool:
        return bool(self.week52_high and self.price >= self.week52_high * 0.999)

    @property
    def at_52w_low(self) -> bool:
        return bool(self.week52_low and self.price <= self.week52_low * 1.001)


class MarketContext(BaseModel):
    """Benchmark and sector baselines, so idiosyncratic moves can be separated out."""

    benchmark: str = "SPY"
    benchmark_pct: float = 0.0
    sector_pct: dict[str, float] = Field(default_factory=dict)
    as_of: datetime | None = None


class Mover(BaseModel):
    """A ticker that cleared screening, with the reasons it did."""

    symbol: str
    name: str
    sector_etf: str | None = None
    quote: Quote
    beta: float | None = None
    residual_pct: float | None = Field(
        default=None, description="Move not explained by the benchmark, in percent"
    )
    sector_pct: float | None = None
    triggers: list[str] = Field(default_factory=list)

    @property
    def direction(self) -> str:
        return "up" if self.quote.pct_change >= 0 else "down"


class Article(BaseModel):
    url: str
    title: str
    source: str = ""
    published_at: datetime | None = None
    summary: str = ""
    text: str = ""
    previously_reported: bool = False

    @property
    def body(self) -> str:
        """Best available text: extracted article, else the RSS summary."""
        return self.text or self.summary


# --------------------------------------------------------------------------------------
# LLM output contracts
# --------------------------------------------------------------------------------------


class TriageSelection(BaseModel):
    idx: int = Field(description="Index of the headline in the supplied list")
    relevance: Relevance
    why: str = Field(default="", description="Under 12 words")


class TriageResult(BaseModel):
    """Sub-agent 1: pick the headlines that could plausibly explain the move."""

    selected: list[TriageSelection] = Field(default_factory=list)
    no_material_news: bool = False


class EvidenceItem(BaseModel):
    claim: str
    source_idx: int = Field(description="Index of the supporting article")


class AnalystResult(BaseModel):
    """Sub-agent 2: attribute the move to a cause, citing supplied articles."""

    headline: str = Field(description="One sentence, 15 words or fewer")
    narrative: str = Field(description="2-4 sentences")
    driver: Driver
    confidence: Confidence
    evidence: list[EvidenceItem] = Field(default_factory=list)
    watch_next: str = ""


class VerifierDrop(BaseModel):
    symbol: str
    claim_idx: int
    reason: str = ""


class VerifierResult(BaseModel):
    """Reflection pass: strike claims the cited article does not support."""

    drops: list[VerifierDrop] = Field(default_factory=list)


class ExecutiveSummary(BaseModel):
    summary: str = Field(description="3-5 sentences")
    themes: list[str] = Field(default_factory=list)
    top_story: str = ""


# --------------------------------------------------------------------------------------
# Assembled per-ticker result
# --------------------------------------------------------------------------------------


class Finding(BaseModel):
    """Everything the report needs about one mover."""

    mover: Mover
    articles: list[Article] = Field(default_factory=list)
    analysis: AnalystResult | None = None
    dropped_claims: list[str] = Field(default_factory=list)
    degraded: bool = Field(
        default=False, description="LLM analysis unavailable; render data only"
    )
    degraded_reason: str = ""

    @property
    def cited_articles(self) -> list[tuple[str, Article]]:
        """(claim, article) pairs for evidence that survived verification."""
        if not self.analysis:
            return []
        pairs: list[tuple[str, Article]] = []
        for item in self.analysis.evidence:
            if 0 <= item.source_idx < len(self.articles):
                pairs.append((item.claim, self.articles[item.source_idx]))
        return pairs
