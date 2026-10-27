"""Full-text extraction for candidate articles.

Extraction is strictly best-effort. Paywalls, consent walls and Google News redirect
stubs all defeat it routinely, and that is fine: the Article falls back to its RSS
summary, and the analyst agent is told which entries are headline-only so it can weigh
them accordingly. A failed fetch must never fail a run.
"""

from __future__ import annotations

import asyncio
import logging

import httpx
import trafilatura

from fi_agent.config import NewsSettings
from fi_agent.data.store import Store
from fi_agent.schemas import Article

log = logging.getLogger(__name__)

# Domains that reliably serve a consent or paywall interstitial instead of content.
SKIP_DOMAINS = ("news.google.com", "wsj.com", "ft.com", "bloomberg.com", "barrons.com")


def _extract(html: str) -> str:
    try:
        text = trafilatura.extract(
            html, include_comments=False, include_tables=False, no_fallback=False
        )
    except Exception as exc:  # trafilatura raises a variety of parser errors
        log.debug("extraction failed: %s", exc)
        return ""
    return (text or "").strip()


async def _fetch_one(
    client: httpx.AsyncClient,
    article: Article,
    store: Store | None,
    char_limit: int,
    semaphore: asyncio.Semaphore,
) -> None:
    if any(domain in article.url for domain in SKIP_DOMAINS):
        return

    if store is not None:
        cached = store.get_article_text(article.url)
        if cached is not None:
            article.text = cached
            return

    async with semaphore:
        try:
            response = await client.get(article.url)
            response.raise_for_status()
            html = response.text
        except (httpx.HTTPError, httpx.InvalidURL) as exc:
            log.debug("article fetch failed %s: %s", article.url, exc)
            return

    text = await asyncio.to_thread(_extract, html)
    if not text:
        return

    article.text = text[:char_limit]
    if store is not None:
        store.put_article_text(article.url, article.text)


async def hydrate(
    articles: list[Article],
    settings: NewsSettings,
    store: Store | None = None,
) -> list[Article]:
    """Populate `.text` on as many articles as possible, in parallel.

    Network fan-out is genuinely concurrent here, unlike LLM calls, so this is where
    parallelism actually buys time.
    """
    if not articles:
        return articles

    semaphore = asyncio.Semaphore(settings.fetch_concurrency)
    headers = {"User-Agent": settings.user_agent}
    async with httpx.AsyncClient(
        timeout=settings.http_timeout_s, headers=headers, follow_redirects=True
    ) as client:
        await asyncio.gather(
            *(
                _fetch_one(client, article, store, settings.article_char_limit, semaphore)
                for article in articles
            )
        )

    hydrated = sum(1 for a in articles if a.text)
    log.debug("hydrated %d/%d articles", hydrated, len(articles))
    return articles
