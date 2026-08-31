"""The supervisor graph.

Control flow is deliberately deterministic. The model is never asked what to do next -
it is asked three narrow questions at three fixed points, and Python decides everything
else. On a 4-bit 7B model an open-ended ReAct loop wanders; a fixed graph does not.

Concurrency note: the Ollama server serialises inference, so the `Send` fan-out below
does not make the LLM work faster. It exists so that the *network* half of each ticker's
work - RSS fetches and article extraction - overlaps across tickers, which is where the
real time savings are. LLM calls are pushed to worker threads so they never block that
I/O.
"""

from __future__ import annotations

import asyncio
import logging
import operator
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from fi_agent.agents.analyst import analyze
from fi_agent.agents.synthesizer import fallback_summary, synthesize
from fi_agent.agents.triage import triage
from fi_agent.agents.verifier import verify
from fi_agent.analysis.screen import build_market_context, screen
from fi_agent.config import Settings, Watchlist
from fi_agent.data.article import hydrate
from fi_agent.data.market import fetch_quotes
from fi_agent.data.news import gather_headlines
from fi_agent.data.store import Store
from fi_agent.llm import LLMClient
from fi_agent.schemas import (
    ExecutiveSummary,
    Finding,
    MarketContext,
    Mover,
    Quote,
)

log = logging.getLogger(__name__)


@dataclass
class Deps:
    """Everything the nodes need that is not run state."""

    settings: Settings
    watchlist: Watchlist
    client: LLMClient | None          # None in --no-llm mode
    store: Store | None
    run_id: str

    @property
    def llm_enabled(self) -> bool:
        return self.client is not None


class RunState(TypedDict, total=False):
    run_id: str
    started_at: datetime
    deadline: float
    quotes: dict[str, Quote]
    context: MarketContext
    movers: list[Mover]
    quiet: list[Mover]
    findings: Annotated[list[Finding], operator.add]
    errors: Annotated[list[str], operator.add]
    summary: ExecutiveSummary


class TickerTask(TypedDict):
    """Payload handed to each fanned-out ticker node."""

    mover: Mover
    context: MarketContext
    deadline: float


def build_graph(deps: Deps):  # noqa: C901 - a graph definition reads better in one place
    settings = deps.settings

    # -- node: collect market data and screen ------------------------------------------

    async def collect(state: RunState) -> RunState:
        symbols = list(deps.watchlist.symbols())
        symbols.append(settings.screening.benchmark)
        symbols.extend(t.sector_etf for t in deps.watchlist.tickers if t.sector_etf)

        quotes, frame, failed = await asyncio.to_thread(fetch_quotes, symbols)
        errors = [f"no market data for {s}" for s in failed]

        context = build_market_context(quotes, deps.watchlist, settings.screening)
        movers, quiet = screen(
            quotes,
            deps.watchlist,
            context,
            settings.screening,
            frame,
            settings.run.max_movers,
        )
        log.info(
            "screened %d watchlist names: %d movers, %d quiet",
            len(deps.watchlist.tickers), len(movers), len(quiet),
        )

        if deps.store is not None:
            for quote in quotes.values():
                deps.store.save_snapshot(deps.run_id, quote)

        return {
            "quotes": quotes,
            "context": context,
            "movers": movers,
            "quiet": quiet,
            "errors": errors,
        }

    # -- node: one mover, end to end ---------------------------------------------------

    async def investigate(task: TickerTask) -> RunState:
        mover: Mover = task["mover"]
        context: MarketContext = task["context"]
        ticker = deps.watchlist.get(mover.symbol)

        try:
            articles = await gather_headlines(
                mover.symbol,
                ticker.display_name() if ticker else mover.name,
                settings.news,
                aliases=ticker.aliases if ticker else None,
            )
        except Exception as exc:
            log.warning("%s: news gathering failed: %s", mover.symbol, exc)
            return {
                "findings": [
                    Finding(
                        mover=mover,
                        degraded=True,
                        degraded_reason=f"news gathering failed: {exc}",
                    )
                ],
                "errors": [f"{mover.symbol}: news gathering failed"],
            }

        if deps.store is not None:
            for article in articles:
                article.previously_reported = deps.store.was_cited(article.url, mover.symbol)

        if not deps.llm_enabled:
            return {"findings": [Finding(mover=mover, articles=articles[:4], degraded=False)]}

        if time.monotonic() > task["deadline"]:
            log.warning("%s: past run budget, skipping analysis", mover.symbol)
            return {
                "findings": [
                    Finding(
                        mover=mover,
                        articles=articles[:4],
                        degraded=True,
                        degraded_reason="run time budget exhausted",
                    )
                ],
                "errors": [f"{mover.symbol}: skipped, run budget exhausted"],
            }

        assert deps.client is not None
        client = deps.client

        # Triage first so only the shortlist pays for full-text extraction.
        selected = await asyncio.to_thread(
            triage, client, mover, context, articles, settings.news.max_articles_per_ticker
        )
        if selected:
            await hydrate(selected, settings.news, deps.store)

        analysis = await asyncio.to_thread(analyze, client, mover, context, selected)

        if analysis is None:
            return {
                "findings": [
                    Finding(
                        mover=mover,
                        articles=selected,
                        degraded=True,
                        degraded_reason="model did not return valid analysis",
                    )
                ],
                "errors": [f"{mover.symbol}: analysis unavailable"],
            }

        return {"findings": [Finding(mover=mover, articles=selected, analysis=analysis)]}

    # -- node: verification ------------------------------------------------------------

    async def verification(state: RunState) -> RunState:
        findings = state.get("findings", [])
        if not deps.llm_enabled or not findings:
            return {}
        assert deps.client is not None
        await asyncio.to_thread(verify, deps.client, findings)
        return {}

    # -- node: executive summary -------------------------------------------------------

    async def summarise(state: RunState) -> RunState:
        findings = state.get("findings", [])
        context = state.get("context") or MarketContext()

        if not deps.llm_enabled:
            return {"summary": fallback_summary(findings, context)}

        assert deps.client is not None
        summary = await asyncio.to_thread(synthesize, deps.client, findings, context)

        if deps.store is not None:
            for finding in findings:
                for _, article in finding.cited_articles:
                    deps.store.mark_cited(article.url, finding.mover.symbol, deps.run_id)

        return {"summary": summary}

    # -- fan-out ------------------------------------------------------------------------

    def dispatch(state: RunState) -> list[Send] | str:
        movers = state.get("movers", [])
        if not movers:
            return "summarise"
        deadline = state.get("deadline", time.monotonic() + settings.run.budget_seconds)
        return [
            Send("investigate", {"mover": m, "context": state["context"], "deadline": deadline})
            for m in movers
        ]

    graph = StateGraph(RunState)
    graph.add_node("collect", collect)
    # `investigate` is a Send target, so its input is a TickerTask rather than RunState.
    # LangGraph supports that at runtime but its node type does not model it.
    graph.add_node("investigate", investigate)  # type: ignore[arg-type]
    graph.add_node("verification", verification)
    graph.add_node("summarise", summarise)

    graph.add_edge(START, "collect")
    graph.add_conditional_edges("collect", dispatch, ["investigate", "summarise"])
    graph.add_edge("investigate", "verification")
    graph.add_edge("verification", "summarise")
    graph.add_edge("summarise", END)
    return graph.compile()


async def run_pipeline(deps: Deps) -> RunState:
    """Execute one full run and return the final state."""
    compiled = build_graph(deps)
    initial: RunState = {
        "run_id": deps.run_id,
        "started_at": datetime.now(UTC),
        "deadline": time.monotonic() + deps.settings.run.budget_seconds,
        "findings": [],
        "errors": [],
    }
    # Fan-out width is bounded by max_movers, but recursion_limit must still allow one
    # step per dispatched branch plus the fixed nodes.
    result = await compiled.ainvoke(
        initial, config={"recursion_limit": deps.settings.run.max_movers * 2 + 10}
    )
    return result
