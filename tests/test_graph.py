"""The whole pipeline, offline.

Market data, news and the model are all stubbed, so this exercises the real graph
wiring - fan-out, reducers, degradation paths - in milliseconds.
"""

from __future__ import annotations

import asyncio

import pandas as pd
import pytest

from fi_agent.agents import graph as graph_mod
from fi_agent.agents.graph import Deps, run_pipeline
from fi_agent.schemas import Article, TriageResult

from .conftest import FakeLLM, make_quote


@pytest.fixture
def stub_data(monkeypatch):
    """Replace the two network boundaries with deterministic doubles."""
    quotes = {
        "NVDA": make_quote("NVDA", 217.55, 227.98, volume=1_500_000.0),
        "AAPL": make_quote("AAPL", 319.70, 314.58),
        "GOOGL": make_quote("GOOGL", 346.59, 340.66),
        "SPY": make_quote("SPY", 769.35, 771.10),
        "SMH": make_quote("SMH", 553.11, 573.00),
        "XLK": make_quote("XLK", 100.0, 100.0),
        "XLC": make_quote("XLC", 101.4, 100.0),
    }

    def fake_fetch(symbols):
        return {s: quotes[s] for s in symbols if s in quotes}, pd.DataFrame(), []

    async def fake_headlines(symbol, name, settings, now=None, aliases=None):
        return [
            Article(url=f"https://example.com/{symbol}", title=f"{name} moves", source="Test",
                    text="Body text explaining the move."),
        ]

    async def fake_hydrate(articles, settings, store=None):
        return articles

    monkeypatch.setattr(graph_mod, "fetch_quotes", fake_fetch)
    monkeypatch.setattr(graph_mod, "gather_headlines", fake_headlines)
    monkeypatch.setattr(graph_mod, "hydrate", fake_hydrate)
    return quotes


def run(deps) -> dict:
    return asyncio.run(run_pipeline(deps))


def test_pipeline_produces_a_finding_per_mover(settings, watchlist, stub_data, fake_llm):
    deps = Deps(settings=settings, watchlist=watchlist, client=fake_llm, store=None, run_id="t")
    state = run(deps)

    symbols = {f.mover.symbol for f in state["findings"]}
    assert "NVDA" in symbols, "a -4.6% move must be flagged"
    assert len(state["findings"]) == len(state["movers"])
    assert state["summary"].summary


def test_no_llm_mode_makes_no_calls(settings, watchlist, stub_data, fake_llm):
    deps = Deps(settings=settings, watchlist=watchlist, client=None, store=None, run_id="t")
    state = run(deps)
    assert state["findings"], "data-only runs still produce findings"
    assert all(f.analysis is None for f in state["findings"])
    assert state["summary"].summary, "a deterministic summary is still written"


def test_llm_call_budget_is_two_per_mover_plus_two(settings, watchlist, stub_data, fake_llm):
    deps = Deps(settings=settings, watchlist=watchlist, client=fake_llm, store=None, run_id="t")
    state = run(deps)
    n_movers = len(state["movers"])
    # triage + analyst per mover, then one verifier and one synthesizer.
    assert fake_llm.calls == n_movers * 2 + 2
    assert fake_llm.labels.count("verifier") == 1
    assert fake_llm.labels.count("synthesizer") == 1


def test_analysis_failure_degrades_only_that_ticker(settings, watchlist, stub_data):
    class HalfBroken(FakeLLM):
        def structured(self, schema, system, user, label=""):
            if schema.__name__ == "AnalystResult" and "NVDA" in label:
                self.calls += 1
                self.labels.append(label)
                return None
            return super().structured(schema, system, user, label)

    deps = Deps(
        settings=settings, watchlist=watchlist, client=HalfBroken(), store=None, run_id="t"
    )
    state = run(deps)

    by_symbol = {f.mover.symbol: f for f in state["findings"]}
    assert by_symbol["NVDA"].degraded
    assert "valid analysis" in by_symbol["NVDA"].degraded_reason
    others = [f for s, f in by_symbol.items() if s != "NVDA"]
    assert all(not f.degraded for f in others), "one failure must not poison the run"
    assert state["summary"].summary


def test_news_failure_degrades_gracefully(settings, watchlist, stub_data, fake_llm, monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("feed down")

    monkeypatch.setattr(graph_mod, "gather_headlines", boom)
    deps = Deps(settings=settings, watchlist=watchlist, client=fake_llm, store=None, run_id="t")
    state = run(deps)

    assert all(f.degraded for f in state["findings"])
    assert state["errors"], "the failure is surfaced in diagnostics"


def test_no_material_news_never_claims_company_causation(settings, watchlist, stub_data):
    """When triage finds nothing, the report may still attribute a move to sector or
    macro forces - those come from price arithmetic - but never to company news, and
    never with confidence."""
    client = FakeLLM(TriageResult=TriageResult(selected=[], no_material_news=True))
    deps = Deps(settings=settings, watchlist=watchlist, client=client, store=None, run_id="t")
    state = run(deps)

    for finding in state["findings"]:
        assert finding.articles == []
        assert finding.analysis.driver in {"sector", "macro", "no_identified_catalyst"}
        assert finding.analysis.confidence == "low"
        assert finding.analysis.evidence == []


def test_exhausted_budget_skips_analysis(settings, watchlist, stub_data, fake_llm):
    settings.run.budget_seconds = -1  # already past the deadline
    deps = Deps(settings=settings, watchlist=watchlist, client=fake_llm, store=None, run_id="t")
    state = run(deps)

    assert all(f.degraded for f in state["findings"])
    assert all("budget" in f.degraded_reason for f in state["findings"])


def test_quiet_names_are_retained(settings, watchlist, stub_data, fake_llm):
    deps = Deps(settings=settings, watchlist=watchlist, client=fake_llm, store=None, run_id="t")
    state = run(deps)
    covered = {f.mover.symbol for f in state["findings"]} | {m.symbol for m in state["quiet"]}
    assert covered == set(watchlist.symbols()), "every watchlist name appears somewhere"
