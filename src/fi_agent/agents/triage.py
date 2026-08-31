"""Triage sub-agent: pick the headlines that could explain a move. One LLM call."""

from __future__ import annotations

import logging

from fi_agent.agents.prompts import TRIAGE_SYSTEM, triage_user
from fi_agent.llm import LLMClient
from fi_agent.schemas import Article, MarketContext, Mover, TriageResult

log = logging.getLogger(__name__)

RELEVANCE_RANK = {"high": 0, "medium": 1, "low": 2}


def triage(
    client: LLMClient,
    mover: Mover,
    context: MarketContext,
    articles: list[Article],
    max_articles: int,
) -> list[Article]:
    """Return the subset of articles worth full analysis, best first.

    Indices the model invents are discarded silently rather than retried - a bad index
    is cheap to drop and expensive to re-ask about on a serialised server.
    """
    if not articles:
        return []

    result = client.structured(
        TriageResult,
        TRIAGE_SYSTEM,
        triage_user(mover, context, articles),
        label=f"triage:{mover.symbol}",
    )

    if result is None:
        # Degrade to recency order rather than dropping the ticker entirely.
        log.warning("%s: triage unavailable, falling back to most recent", mover.symbol)
        return articles[:max_articles]

    if result.no_material_news:
        log.info("%s: triage found no material news", mover.symbol)
        return []

    valid = [s for s in result.selected if 0 <= s.idx < len(articles)]
    dropped = len(result.selected) - len(valid)
    if dropped:
        log.warning("%s: triage returned %d out-of-range indices", mover.symbol, dropped)

    valid.sort(key=lambda s: RELEVANCE_RANK.get(s.relevance, 3))

    seen: set[int] = set()
    chosen: list[Article] = []
    for selection in valid:
        if selection.idx in seen:
            continue
        seen.add(selection.idx)
        chosen.append(articles[selection.idx])
        if len(chosen) >= max_articles:
            break
    return chosen
