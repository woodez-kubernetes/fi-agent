"""News merging, relevance filtering and deduplication."""

from __future__ import annotations

import asyncio
from datetime import timedelta

import httpx
import pytest

from fi_agent.config import NewsSettings
from fi_agent.data import news
from fi_agent.data.news import (
    _needles,
    canonical_url,
    deduplicate,
    filter_relevant,
    parse_feed,
    relevance_score,
)
from fi_agent.schemas import Article

from .conftest import NOW

FEED = b"""<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item>
    <title>Nvidia slides as chip selloff deepens - Reuters</title>
    <link>https://example.com/story?utm_source=rss&amp;id=7</link>
    <pubDate>Sat, 30 Aug 2026 17:00:00 GMT</pubDate>
    <description>&lt;p&gt;Chip stocks fell sharply.&lt;/p&gt;</description>
  </item>
  <item>
    <title>Unrelated market wrap</title>
    <link>https://example.com/wrap</link>
    <pubDate>Sat, 30 Aug 2026 16:00:00 GMT</pubDate>
  </item>
</channel></rss>"""


def test_parse_feed_extracts_fields_and_strips_html():
    articles = parse_feed(FEED, "Google News")
    assert len(articles) == 2
    first = articles[0]
    assert first.source == "Reuters"
    assert first.title == "Nvidia slides as chip selloff deepens", "publisher suffix stripped"
    assert first.summary == "Chip stocks fell sharply."
    assert first.published_at is not None and first.published_at.tzinfo is not None


def test_parse_feed_tolerates_garbage():
    assert parse_feed(b"not xml at all", "X") == []


def test_canonical_url_strips_tracking_and_case():
    assert canonical_url("https://WWW.Example.com/a/?utm_source=x&id=7#frag") == (
        "https://example.com/a?id=7"
    )


def test_canonical_url_keeps_meaningful_query():
    assert "id=7" in canonical_url("https://example.com/a?id=7")


def test_needles_drop_corporate_suffix():
    needles = _needles("AMD", "Advanced Micro Devices Inc")
    assert "amd" in needles
    assert "advanced micro devices" in needles


def test_relevance_prefers_headline_mentions():
    needles = _needles("NVDA", "NVIDIA")
    in_title = Article(url="u1", title="Nvidia falls hard", summary="")
    in_summary = Article(url="u2", title="Chip wrap", summary="Nvidia led decliners")
    unrelated = Article(url="u3", title="CVS Health rallies", summary="Pharmacy news")

    assert relevance_score(in_title, needles) == 2
    assert relevance_score(in_summary, needles) == 1
    assert relevance_score(unrelated, needles) == 0


def test_relevance_uses_word_boundaries():
    """Without boundaries, META matches 'metaverse' and AMD matches 'AMDOCS'."""
    needles = _needles("META", "Meta Platforms")
    assert relevance_score(Article(url="u", title="The metaverse cools"), needles) == 0
    assert relevance_score(Article(url="u", title="Meta buys a studio"), needles) == 2


def test_aliases_catch_press_names():
    needles = _needles("GOOGL", "Alphabet", aliases=["Google"])
    assert relevance_score(Article(url="u", title="Google ad revenue climbs"), needles) == 2


def test_filter_relevant_drops_noise_and_ranks_headlines_first():
    articles = [
        Article(url="u1", title="Market wrap", summary="Nvidia led decliners",
                published_at=NOW),
        Article(url="u2", title="Nvidia slides", summary="", published_at=NOW - timedelta(hours=2)),
        Article(url="u3", title="CVS Health rallies", summary="", published_at=NOW),
    ]
    kept = filter_relevant(articles, "NVDA", "NVIDIA")
    assert [a.url for a in kept] == ["u2", "u1"], "headline match outranks fresher summary match"


def test_deduplicate_by_canonical_url():
    articles = [
        Article(url="https://example.com/a?utm_source=rss", title="One"),
        Article(url="https://example.com/a", title="Something else entirely"),
    ]
    assert len(deduplicate(articles)) == 1


def test_deduplicate_catches_syndicated_rewrites():
    articles = [
        Article(url="https://a.com/1", title="Nvidia slides as chip selloff deepens"),
        Article(url="https://b.com/2", title="Nvidia slides as the chip selloff deepens"),
        Article(url="https://c.com/3", title="Apple unveils a new laptop"),
    ]
    kept = deduplicate(articles, fuzzy_threshold=88)
    assert len(kept) == 2


def test_article_body_falls_back_to_summary():
    assert Article(url="u", title="t", summary="s").body == "s"
    assert Article(url="u", title="t", summary="s", text="full").body == "full"


# --------------------------------------------------------------------------------------
# Yahoo rate-limit cool-down. HTTP is served by httpx.MockTransport; nothing goes out.
# --------------------------------------------------------------------------------------

class FakeFeeds:
    """Yahoo answers 429 while `yahoo_blocked`; Google always serves FEED."""

    def __init__(self) -> None:
        self.yahoo_blocked = True
        self.hits = {"yahoo": 0, "google": 0}

    def handler(self, request: httpx.Request) -> httpx.Response:
        if "yahoo" in request.url.host:
            self.hits["yahoo"] += 1
            if self.yahoo_blocked:
                return httpx.Response(429, text="Too Many Requests")
            return httpx.Response(200, content=FEED)
        self.hits["google"] += 1
        return httpx.Response(200, content=FEED)


@pytest.fixture
def feeds(monkeypatch):
    fake = FakeFeeds()
    real_client = httpx.AsyncClient
    clock = {"now": 1000.0}

    async def no_sleep(_s):
        return None

    monkeypatch.setattr(
        news.httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(fake.handler), **kw),
    )
    monkeypatch.setattr(news.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(news, "_yfinance_news", lambda _symbol: [])
    monkeypatch.setattr(news, "_clock", lambda: clock["now"])
    monkeypatch.setattr(news, "_yahoo_paused_until", 0.0)
    fake.clock = clock
    return fake


async def _gather():
    return await news.gather_headlines("NVDA", "Nvidia", NewsSettings(), now=NOW)


async def test_persistent_429_pauses_yahoo_and_keeps_google(feeds, caplog):
    articles = await _gather()

    assert feeds.hits["yahoo"] == 3  # the normal retries, once
    assert news.yahoo_paused()
    assert any(a.source == "Reuters" for a in articles), "Google News still used"
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert warnings == [
        "Yahoo RSS is rate-limiting; skipping it for 60 min (Google News still used)"
    ]


async def test_paused_yahoo_is_not_requested(feeds):
    await _gather()
    await _gather()
    assert feeds.hits["yahoo"] == 3
    assert feeds.hits["google"] == 2


async def test_yahoo_is_retried_after_cool_down(feeds):
    await _gather()
    feeds.yahoo_blocked = False
    feeds.clock["now"] += news.YAHOO_COOLDOWN_S + 1
    await _gather()
    assert feeds.hits["yahoo"] == 4
    assert not news.yahoo_paused()


async def test_concurrent_tickers_report_the_block_once(feeds, caplog):
    await asyncio.gather(*(_gather() for _ in range(5)))
    pauses = [r for r in caplog.records if "Yahoo RSS is rate-limiting" in r.getMessage()]
    assert len(pauses) == 1
