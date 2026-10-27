"""News gathering from three free sources, merged and deduplicated.

Sources differ in what they are good for:

* Yahoo Finance RSS and yfinance both link straight at publisher pages, so their
  entries usually yield full article text.
* Google News RSS gives much wider coverage but its links are redirect stubs that
  frequently resist extraction. Its entries are still useful as headlines, which is
  all the triage agent needs to decide relevance.

Nothing here calls an LLM.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import UTC, datetime, timedelta
from urllib.parse import quote_plus, urlparse, urlunparse

import feedparser
import httpx
from dateutil import parser as date_parser
from rapidfuzz import fuzz

from fi_agent.config import NewsSettings
from fi_agent.schemas import Article

log = logging.getLogger(__name__)

YAHOO_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
GOOGLE_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

# Query params that identify the referrer rather than the document.
TRACKING_PARAMS = re.compile(r"^(utm_|fbclid|gclid|ref|ref_src|guccounter|yptr)")


def canonical_url(url: str) -> str:
    """Strip tracking parameters and fragments so the same story dedupes across feeds."""
    try:
        parts = urlparse(url)
    except ValueError:
        return url
    kept = [
        pair
        for pair in parts.query.split("&")
        if pair and not TRACKING_PARAMS.match(pair.split("=")[0])
    ]
    netloc = parts.netloc.lower().removeprefix("www.")
    return urlunparse((parts.scheme, netloc, parts.path.rstrip("/"), "", "&".join(kept), ""))


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = date_parser.parse(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _clean(text: str) -> str:
    """RSS summaries are frequently HTML fragments."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text or "")).strip()


def _entry_source(entry: object, fallback: str) -> str:
    source = getattr(entry, "source", None)
    title = getattr(source, "title", None) if source else None
    if title:
        return str(title)
    # Google News appends " - Publisher" to entry titles.
    entry_title = getattr(entry, "title", "") or ""
    if " - " in entry_title:
        return entry_title.rsplit(" - ", 1)[1].strip()
    return fallback


def parse_feed(content: bytes, fallback_source: str) -> list[Article]:
    """Turn raw RSS bytes into Articles. Never raises - a bad feed yields nothing."""
    try:
        parsed = feedparser.parse(content)
    except Exception as exc:  # feedparser is lenient but not infallible
        log.warning("feed parse failed: %s", exc)
        return []

    articles: list[Article] = []
    for entry in parsed.entries:
        link = getattr(entry, "link", "") or ""
        title = _clean(getattr(entry, "title", "") or "")
        if not link or not title:
            continue
        # Strip the " - Publisher" suffix Google appends, once captured as the source.
        source = _entry_source(entry, fallback_source)
        if fallback_source == "Google News" and title.endswith(f" - {source}"):
            title = title[: -(len(source) + 3)].strip()
        articles.append(
            Article(
                url=link,
                title=title,
                source=source,
                published_at=_parse_date(
                    getattr(entry, "published", None) or getattr(entry, "updated", None)
                ),
                summary=_clean(getattr(entry, "summary", "") or "")[:600],
            )
        )
    return articles


async def _get(client: httpx.AsyncClient, url: str, attempts: int = 3) -> bytes | None:
    """GET with backoff. Yahoo's RSS endpoint rate-limits aggressively and returns 429
    for a burst of ticker requests, so a retry is the difference between having the
    source and not."""
    delay = 1.0
    for attempt in range(attempts):
        try:
            response = await client.get(url)
            if response.status_code == 429 and attempt < attempts - 1:
                await asyncio.sleep(delay)
                delay *= 2
                continue
            response.raise_for_status()
            return response.content
        except (httpx.HTTPError, httpx.InvalidURL) as exc:
            if attempt == attempts - 1:
                log.warning("feed fetch failed %s: %s", url, exc)
                return None
            await asyncio.sleep(delay)
            delay *= 2
    return None


def _yfinance_news(symbol: str) -> list[Article]:
    """yfinance's own news list. Synchronous and occasionally absent, so it is
    wrapped defensively and treated as a bonus source."""
    try:
        import yfinance as yf

        raw = yf.Ticker(symbol).news or []
    except Exception as exc:
        log.warning("yfinance news failed for %s: %s", symbol, exc)
        return []

    articles: list[Article] = []
    for item in raw:
        # yfinance 1.x nests the payload under "content"; older versions were flat.
        content = item.get("content", item) if isinstance(item, dict) else {}
        title = _clean(content.get("title", ""))
        url = ""
        for key in ("canonicalUrl", "clickThroughUrl"):
            value = content.get(key)
            if isinstance(value, dict) and value.get("url"):
                url = value["url"]
                break
        url = url or content.get("link", "") or ""
        if not title or not url:
            continue
        provider = content.get("provider") or {}
        articles.append(
            Article(
                url=url,
                title=title,
                source=(provider.get("displayName") if isinstance(provider, dict) else "")
                or "Yahoo Finance",
                published_at=_parse_date(content.get("pubDate") or content.get("displayTime")),
                summary=_clean(content.get("summary", ""))[:600],
            )
        )
    return articles


def _needles(symbol: str, company_name: str, aliases: list[str] | None = None) -> list[str]:
    """Terms whose presence marks an article as being about this company.

    The company's legal suffix is dropped ("Advanced Micro Devices Inc" -> the first
    words carry the signal) and single-word forms are kept so "Tesla" matches even when
    the feed writes "Tesla, Inc.".
    """
    terms = {symbol.lower()}
    name = re.sub(r"[.,]", " ", (company_name or "")).strip()
    name = re.sub(
        r"\b(inc|corp|corporation|co|ltd|plc|holdings|group|company)\b", "", name, flags=re.I
    )
    name = re.sub(r"\s+", " ", name).strip()
    if name:
        terms.add(name.lower())
        first = name.split()[0].lower()
        # Only useful when distinctive; "advanced" or "international" would match anything.
        if len(first) >= 4:
            terms.add(first)
    terms.update(a.lower() for a in (aliases or []) if a)
    return sorted(terms)


def relevance_score(article: Article, needles: list[str]) -> int:
    """2 if the company is named in the headline, 1 if only in the summary, else 0.

    Both Yahoo's and yfinance's per-ticker feeds mix in unrelated market stories - a
    request for NVDA news returns pieces on CVS and Bitcoin. Without this filter that
    noise reaches the triage agent, which then has to reason about why a CVS story is
    in front of it. Screening it out deterministically is both cheaper and more reliable.
    """
    title = article.title.lower()
    summary = article.summary.lower()
    for needle in needles:
        # Word-boundary match so "AMD" does not hit "AMDOCS" and "META" not "metaverse".
        pattern = rf"\b{re.escape(needle)}\b"
        if re.search(pattern, title):
            return 2
        if re.search(pattern, summary):
            return 1
    return 0


def filter_relevant(
    articles: list[Article], symbol: str, company_name: str, aliases: list[str] | None = None
) -> list[Article]:
    """Keep only articles that actually name the company, headline matches first."""
    needles = _needles(symbol, company_name, aliases)
    scored = [(relevance_score(a, needles), a) for a in articles]
    kept = [(score, a) for score, a in scored if score > 0]
    kept.sort(
        key=lambda pair: (
            pair[0],
            pair[1].published_at or datetime.min.replace(tzinfo=UTC),
        ),
        reverse=True,
    )
    return [article for _, article in kept]


def deduplicate(articles: list[Article], fuzzy_threshold: int = 88) -> list[Article]:
    """Drop repeats: exact canonical-URL matches first, then near-identical titles.

    Syndicated wire stories appear under several publishers with slightly different
    headlines, which is what the fuzzy pass catches.
    """
    unique: list[Article] = []
    seen_urls: set[str] = set()
    for article in articles:
        key = canonical_url(article.url)
        if key in seen_urls:
            continue
        if any(
            fuzz.token_set_ratio(article.title.lower(), kept.title.lower()) >= fuzzy_threshold
            for kept in unique
        ):
            continue
        seen_urls.add(key)
        unique.append(article)
    return unique


async def gather_headlines(
    symbol: str,
    company_name: str,
    settings: NewsSettings,
    now: datetime | None = None,
    aliases: list[str] | None = None,
) -> list[Article]:
    """Collect, filter and dedupe headlines for one ticker."""
    query = quote_plus(f"{company_name or symbol} stock")
    urls = [
        (YAHOO_RSS.format(symbol=symbol), "Yahoo Finance"),
        (GOOGLE_RSS.format(query=query), "Google News"),
    ]

    headers = {"User-Agent": settings.user_agent}
    async with httpx.AsyncClient(
        timeout=settings.http_timeout_s, headers=headers, follow_redirects=True
    ) as client:
        payloads = await asyncio.gather(*(_get(client, url) for url, _ in urls))

    collected: list[Article] = []
    for payload, (_, source) in zip(payloads, urls, strict=True):
        if payload:
            collected.extend(parse_feed(payload, source))

    collected.extend(await asyncio.to_thread(_yfinance_news, symbol))

    cutoff = (now or datetime.now(UTC)) - timedelta(hours=settings.lookback_hours)
    fresh = [a for a in collected if a.published_at is None or a.published_at >= cutoff]

    relevant = filter_relevant(fresh, symbol, company_name, aliases)
    log.debug(
        "%s: %d collected -> %d fresh -> %d relevant", symbol, len(collected), len(fresh),
        len(relevant),
    )
    return deduplicate(relevant, settings.fuzzy_title_threshold)[
        : settings.max_headlines_per_ticker
    ]
