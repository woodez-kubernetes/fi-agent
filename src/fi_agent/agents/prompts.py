"""Prompt construction.

The prompts here are written against observed failure modes of qwen2.5:7b, not against
an idealised model. Three lessons shaped them:

1. Asked to classify a broad semiconductor selloff, the model chose "analyst_action".
   Enum values therefore carry an explicit one-line definition and a decision rule, and
   the *arithmetic* of sector-versus-idiosyncratic is done in Python and handed over as
   a stated fact rather than left for the model to infer.
2. Asked for a headline of "15 words or fewer" it returned a 23-word sentence copied
   verbatim from an article. Length limits are restated as hard instructions and
   enforced in post-processing.
3. Asked for evidence it produced claims that merely restated the price move, and an
   article index of -1. The prompt bans restating numbers and the caller validates
   indices deterministically.
"""

from __future__ import annotations

from fi_agent.schemas import Article, MarketContext, Mover

NO_ADVICE = (
    "Never recommend buying or selling, never give a price target, and never predict "
    "future prices. Describe only what happened and what the sources say caused it."
)


def classify_move(mover: Mover, context: MarketContext) -> str:
    """Decide in Python whether a move is market-, sector- or company-driven.

    The model is poor at this comparison and good at reading a stated conclusion, so
    the conclusion is computed here and asserted in the prompt.
    """
    move = mover.quote.pct_change
    if abs(move) < 0.01:
        return "flat"

    sector = mover.sector_pct
    residual = mover.residual_pct

    if sector is not None and abs(sector) >= abs(move) * 0.6 and sector * move > 0:
        return "sector-wide"
    if residual is not None and abs(residual) < abs(move) * 0.4:
        return "market-wide"
    return "company-specific"


def move_brief(mover: Mover, context: MarketContext) -> str:
    """The numeric facts, rendered as prose. Every figure comes from Python."""
    quote = mover.quote
    lines = [
        f"{mover.symbol} ({mover.name}) is {quote.price:.2f} {quote.currency}, "
        f"{quote.pct_change:+.2f}% versus the previous close of {quote.prev_close:.2f}.",
        f"The {context.benchmark} benchmark is {context.benchmark_pct:+.2f}% on the session.",
    ]
    if mover.sector_etf and mover.sector_pct is not None:
        lines.append(
            f"Its sector ETF {mover.sector_etf} is {mover.sector_pct:+.2f}%."
        )
    if mover.residual_pct is not None and mover.beta is not None:
        lines.append(
            f"After adjusting for a beta of {mover.beta:.2f}, {mover.residual_pct:+.2f}% "
            f"of the move is not explained by the broad market."
        )
    ratio = quote.volume_ratio
    if ratio is not None:
        lines.append(f"Volume is running at {ratio:.1f}x its 30-day average.")
    if quote.at_52w_high:
        lines.append("The stock is at a 52-week high.")
    elif quote.at_52w_low:
        lines.append("The stock is at a 52-week low.")

    classification = classify_move(mover, context)
    lines.append(
        {
            "sector-wide": "ASSESSMENT: the sector moved in line with the stock, so this "
            "is mostly a sector-wide move, not a company-specific one.",
            "market-wide": "ASSESSMENT: the broad market explains most of this move.",
            "company-specific": "ASSESSMENT: most of this move is specific to the company "
            "and is not explained by the market or sector.",
            "flat": "ASSESSMENT: the stock is essentially unchanged.",
        }[classification]
    )
    return "\n".join(lines)


def _when(article: Article) -> str:
    return article.published_at.strftime("%b %d %H:%M UTC") if article.published_at else "undated"


def format_headlines(articles: list[Article]) -> str:
    entries = []
    for idx, article in enumerate(articles):
        entries.append(f"[{idx}] ({article.source}, {_when(article)}) {article.title}")
    return "\n".join(entries)


def format_articles(articles: list[Article]) -> str:
    blocks = []
    for idx, article in enumerate(articles):
        kind = "FULL TEXT" if article.text else "HEADLINE AND SUMMARY ONLY"
        body = article.body or "(no body text available)"
        blocks.append(
            f"ARTICLE [{idx}] ({article.source}, {_when(article)}) - {kind}\n"
            f"TITLE: {article.title}\n{body}"
        )
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------------------
# Triage
# --------------------------------------------------------------------------------------

TRIAGE_SYSTEM = f"""You are a financial news triage filter.

You are given one stock's price move and a numbered list of recent headlines. Select \
only the headlines that could plausibly explain that move, ranked most relevant first.

Rules:
- Select at most 4 headlines. Selecting fewer is better than selecting weak ones.
- Use the index numbers exactly as given. Never invent an index.
- Opinion pieces, valuation musings, "should you buy" articles and listicles are NOT \
explanations of a move. Mark them low relevance or leave them out.
- Concrete events are what matter: earnings, guidance, analyst rating changes, \
products, contracts, regulation, lawsuits, executive changes, supply chain news.
- If nothing in the list plausibly explains the move, set no_material_news to true and \
select nothing. This is a common and perfectly acceptable answer.
- {NO_ADVICE}"""


def triage_user(mover: Mover, context: MarketContext, articles: list[Article]) -> str:
    return (
        f"{move_brief(mover, context)}\n\n"
        f"HEADLINES:\n{format_headlines(articles)}\n\n"
        f"Which of these headlines could explain the move? Return JSON only."
    )


# --------------------------------------------------------------------------------------
# Analyst
# --------------------------------------------------------------------------------------

ANALYST_SYSTEM = f"""You are a financial news analyst. You explain why one stock moved, \
using only the articles supplied to you.

Choose `driver` using these definitions, in this order of preference:
- "earnings": the company reported results or changed its own guidance.
- "analyst_action": a named brokerage upgraded, downgraded or changed a price target.
- "company_news": any other company-specific event (product, contract, lawsuit, \
regulation, executive change, supply agreement).
- "sector": the whole industry moved together and no company-specific cause is present. \
If the ASSESSMENT line says the move is sector-wide, use this.
- "macro": broad market forces such as rates, inflation or economic data. If the \
ASSESSMENT line says the move is market-wide, use this.
- "no_identified_catalyst": the supplied articles do not explain the move. Use this \
freely; it is a better answer than a guess.

Hard requirements:
- `headline`: at most 15 words. Write it yourself. Do NOT copy a sentence from an article.
- `narrative`: 2 to 4 sentences explaining the likely cause and how well it fits.
- `evidence`: each claim states a CAUSE reported by an article, in your own words. \
Never restate the price move or any percentage; those are already known. \
`source_idx` must be the index of an article shown above, and must be 0 or greater. \
If no article supports a claim, omit the claim.
- `confidence`: "high" only when an article directly reports a company-specific cause. \
"medium" when the link is plausible but indirect. "low" when you are inferring.
- `watch_next`: one short sentence on what would confirm or refute this reading. \
Never a prediction or a recommendation.
- {NO_ADVICE}"""


def analyst_user(mover: Mover, context: MarketContext, articles: list[Article]) -> str:
    if not articles:
        return (
            f"{move_brief(mover, context)}\n\n"
            "No relevant articles were found for this move. Set driver to "
            "'no_identified_catalyst', confidence to 'low', and leave evidence empty. "
            "Return JSON only."
        )
    return (
        f"{move_brief(mover, context)}\n\n"
        f"{format_articles(articles)}\n\n"
        f"Explain why {mover.symbol} moved. Return JSON only."
    )


# --------------------------------------------------------------------------------------
# Verifier
# --------------------------------------------------------------------------------------

VERIFIER_SYSTEM = """You are a fact-checker. For each numbered claim you are given the \
article that was cited as its support.

Drop a claim if the article does not actually state it. Be strict but not pedantic: \
a fair paraphrase is supported; a detail the article never mentions is not.

Return only the claims that should be DROPPED, using their exact claim_idx and symbol. \
If every claim is supported, return an empty drops list."""


def verifier_user(entries: list[tuple[str, int, str, str]]) -> str:
    """entries: (symbol, claim_idx, claim, supporting article excerpt)."""
    blocks = []
    for symbol, claim_idx, claim, excerpt in entries:
        blocks.append(
            f"--- symbol: {symbol}, claim_idx: {claim_idx}\n"
            f"CLAIM: {claim}\n"
            f"CITED ARTICLE: {excerpt}"
        )
    return (
        "\n\n".join(blocks)
        + "\n\nWhich claims are not supported by their cited article? Return JSON only."
    )


# --------------------------------------------------------------------------------------
# Synthesizer
# --------------------------------------------------------------------------------------

SYNTHESIZER_SYSTEM = f"""You write the opening summary of a daily stock watchlist report.

- `summary`: 3 to 5 sentences covering what moved and why, in plain prose. Group names \
that moved for the same reason. Do not list every ticker mechanically.
- `themes`: 2 to 4 short phrases naming the day's common threads, e.g. "semiconductor \
weakness". Not sentences.
- `top_story`: the ticker symbol with the most consequential move, copied exactly from \
the input.
- You may restate percentages that appear in the input, but never invent one.
- {NO_ADVICE}"""


def synthesizer_user(summaries: list[str], benchmark: str, benchmark_pct: float) -> str:
    body = "\n".join(f"- {line}" for line in summaries)
    return (
        f"The {benchmark} benchmark is {benchmark_pct:+.2f}% on the session.\n\n"
        f"Movers:\n{body}\n\nWrite the report summary. Return JSON only."
    )
