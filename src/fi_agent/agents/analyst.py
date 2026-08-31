"""Analyst sub-agent: attribute a move to a cause, with citations. One LLM call."""

from __future__ import annotations

import logging
import re

from fi_agent.agents.prompts import ANALYST_SYSTEM, analyst_user, classify_move
from fi_agent.llm import LLMClient
from fi_agent.schemas import AnalystResult, Article, Driver, MarketContext, Mover

log = logging.getLogger(__name__)

HEADLINE_WORD_LIMIT = 15

# Drivers that assert a cause specific to this company, and therefore need a citation.
COMPANY_DRIVERS: set[Driver] = {"company_news", "earnings", "analyst_action"}

# Drivers established by price arithmetic rather than by reporting.
DATA_BACKED_DRIVER: dict[str, Driver] = {"sector-wide": "sector", "market-wide": "macro"}
# A claim that is really just a restatement of the price move.
PRICE_RESTATEMENT = re.compile(
    # "...fell 4.5%", "...boosting shares by 4.0%"
    r"\b(fell|rose|dropped|gained|climbed|slid|jumped|declined|surged|boost\w*|lift\w*|"
    r"push\w*|sent|drove|rallied|sank|tumbled|plunged|advanced)\b[^.]{0,40}?\b\d+(\.\d+)?\s*%"
    # "a 4.0% drop"
    r"|\b\d+(\.\d+)?\s*%\s+(drop|gain|fall|rise|decline|increase|move|jump|slide|loss)"
    # "shares are up 4%"
    r"|\bshares?\b[^.]{0,30}?\b(up|down)\b\s+\d+(\.\d+)?\s*%",
    re.IGNORECASE,
)


# Percentages describing the business rather than the share price. "Revenue rose 12%"
# is exactly the kind of evidence the report wants, so it must survive the filter.
FUNDAMENTAL_CONTEXT = re.compile(
    r"\b(revenue|sales|earnings|margin|profit|income|growth|bookings|backlog|users?|"
    r"subscribers?|shipments?|capacity|guidance|orders?|yield|output|deliveries)\b",
    re.IGNORECASE,
)


def is_price_restatement(claim: str) -> bool:
    """True when a claim merely repeats the price move already shown in the table.

    Guarded against fundamentals: a percentage attached to revenue or margin is a
    reported business fact, not a restatement of the stock's move, and dropping it
    would discard the strongest evidence a claim can have.
    """
    match = PRICE_RESTATEMENT.search(claim)
    if not match:
        return False
    # Look at the words immediately around the matched percentage.
    start = max(0, match.start() - 40)
    return not FUNDAMENTAL_CONTEXT.search(claim[start : match.end() + 40])


def _trim_headline(text: str) -> str:
    """Enforce the length limit the model ignores.

    qwen2.5:7b routinely returns a full sentence copied from an article here. Truncating
    at a word boundary is better than shipping a 30-word 'headline'.
    """
    words = text.strip().split()
    if len(words) <= HEADLINE_WORD_LIMIT:
        return text.strip()
    return " ".join(words[:HEADLINE_WORD_LIMIT]).rstrip(",;:") + "..."


def _clean_evidence(result: AnalystResult, n_articles: int, symbol: str) -> None:
    """Drop citations that point nowhere and claims that only restate the price.

    Both are observed failure modes: the model has returned source_idx of -1, and
    "claims" such as "NVDA fell 4.57%", which cite an article for a number that came
    from market data.
    """
    kept = []
    for item in result.evidence:
        if not 0 <= item.source_idx < n_articles:
            log.warning("%s: dropping claim citing article %d", symbol, item.source_idx)
            continue
        if is_price_restatement(item.claim):
            log.debug("%s: dropping price-restatement claim", symbol)
            continue
        kept.append(item)
    result.evidence = kept


def _rewrite_prose(result: AnalystResult, mover: Mover, classification: str) -> None:
    """Replace a narrative whose supporting evidence was struck.

    Written from price data only, so it states what is actually known rather than
    repeating an attribution nothing supports.
    """
    move = mover.quote.pct_change
    direction = "rose" if move >= 0 else "fell"
    result.headline = (
        f"{mover.symbol} {direction} {abs(move):.2f}% with no supported catalyst found"
    )

    parts = [
        f"{mover.name} {direction} {abs(move):.2f}% on the session, but no article in "
        f"the news gathered for this run supports a specific cause."
    ]
    if classification == "sector-wide" and mover.sector_etf and mover.sector_pct is not None:
        parts.append(
            f"The move tracked its sector, with {mover.sector_etf} "
            f"{mover.sector_pct:+.2f}%, so it was likely part of a broader shift."
        )
    elif mover.residual_pct is not None:
        parts.append(
            f"{mover.residual_pct:+.2f}% of the move is unexplained by the benchmark, "
            f"so it does not appear to be a market-wide effect."
        )
    result.narrative = " ".join(parts)
    result.watch_next = "Whether reporting emerges that accounts for the move."


def analyze(
    client: LLMClient,
    mover: Mover,
    context: MarketContext,
    articles: list[Article],
) -> AnalystResult | None:
    """Produce the causal reading for one mover, or None if the model failed."""
    result = client.structured(
        AnalystResult,
        ANALYST_SYSTEM,
        analyst_user(mover, context, articles),
        label=f"analyst:{mover.symbol}",
    )
    if result is None:
        return None

    result.headline = _trim_headline(result.headline)
    _clean_evidence(result, len(articles), mover.symbol)

    # The sector/market split is arithmetic over price data, not a judgement call, so it
    # counts as evidence in its own right - it holds even when no article was found.
    classification = classify_move(mover, context)
    data_driver = DATA_BACKED_DRIVER.get(classification)

    if result.driver in COMPANY_DRIVERS and not result.evidence:
        # A company-specific claim with nothing left to cite is not credible. Fall back
        # to whatever the price data supports.
        log.info("%s: company-specific claim lost all its evidence", mover.symbol)
        result.driver = data_driver or "no_identified_catalyst"
        result.confidence = "low"
        # The prose still asserts the cause that was just struck, so it has to go too.
        # Leaving it would put an unsupported claim in the largest text on the card while
        # the chips beside it say there is no identified catalyst.
        _rewrite_prose(result, mover, classification)
    elif result.driver not in COMPANY_DRIVERS:
        # The model is poor at this comparison and Python is not, so Python wins.
        result.driver = data_driver or (
            "no_identified_catalyst" if not articles else result.driver
        )

    if not articles:
        # Whatever the price data implies, with no reporting behind it this is a guess.
        result.confidence = "low"
    elif not result.evidence and result.confidence == "high":
        # A sector or macro reading is supported by price arithmetic, but with no article
        # behind it "high" overclaims.
        result.confidence = "medium"

    return result
