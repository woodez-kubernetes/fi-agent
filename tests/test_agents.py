"""Agent post-processing: the deterministic guards around model output.

These cover the specific failure modes qwen2.5:7b was observed to produce.
"""

from __future__ import annotations

import pytest

from fi_agent.agents.analyst import _trim_headline, analyze, is_price_restatement
from fi_agent.agents.prompts import classify_move, move_brief
from fi_agent.agents.triage import triage
from fi_agent.agents.verifier import verify
from fi_agent.schemas import (
    AnalystResult,
    EvidenceItem,
    Finding,
    TriageResult,
    TriageSelection,
    VerifierDrop,
    VerifierResult,
    downgrade,
)

from .conftest import FakeLLM

# -- headline length --------------------------------------------------------------------


def test_trim_headline_leaves_short_ones_alone():
    assert _trim_headline("Nvidia falls on chip weakness") == "Nvidia falls on chip weakness"


def test_trim_headline_truncates_copied_sentences():
    long = " ".join(f"word{i}" for i in range(30))
    trimmed = _trim_headline(long)
    assert len(trimmed.split()) <= 16 and trimmed.endswith("...")


# -- price restatement ------------------------------------------------------------------


@pytest.mark.parametrize(
    "claim",
    [
        "An analyst lifted the target, boosting Amazon's share price by 4.0%.",
        "NVDA fell 4.57% on the session",
        "Shares are up 4% today",
        "a 4.0% drop followed the announcement",
    ],
)
def test_price_restatements_are_detected(claim):
    assert is_price_restatement(claim)


@pytest.mark.parametrize(
    "claim",
    [
        "The company reported a 12% rise in cloud revenue",
        "Cloud revenue rose 34% year over year",
        "Operating margin expanded to 18%",
        "SK Hynix warned memory shortages will persist through 2030",
        "Morgan Stanley raised its price target to $355",
    ],
)
def test_business_percentages_survive(claim):
    """Fundamentals are the best evidence available; dropping them would be a bug."""
    assert not is_price_restatement(claim)


# -- evidence cleaning ------------------------------------------------------------------


def test_analyze_drops_out_of_range_citations(mover, context, articles):
    """The model has been observed returning source_idx of -1."""
    bad = AnalystResult(
        headline="Chips fall",
        narrative="Sector weakness.",
        driver="sector",
        confidence="high",
        evidence=[
            EvidenceItem(claim="SK Hynix warned on memory", source_idx=0),
            EvidenceItem(claim="Something unsourced", source_idx=-1),
            EvidenceItem(claim="Also unsourced", source_idx=99),
        ],
    )
    client = FakeLLM(AnalystResult=bad)
    result = analyze(client, mover, context, articles)
    assert [e.source_idx for e in result.evidence] == [0]


def test_analyze_with_no_articles_and_no_price_signal_admits_no_catalyst(mover, context):
    # Nothing in the sector or benchmark explains it, and there is no reporting either.
    mover.sector_pct = 0.1
    mover.residual_pct = -4.5
    result = analyze(FakeLLM(), mover, context, [])
    assert result.driver == "no_identified_catalyst"
    assert result.confidence == "low"


def test_sector_arithmetic_stands_in_for_missing_articles(mover, context):
    """A sector-wide move is established by price data, so it survives having no news -
    but with nothing reported behind it, confidence must drop."""
    result = analyze(FakeLLM(), mover, context, [])
    assert result.driver == "sector"
    assert result.confidence == "low"


def test_analyze_downgrades_company_claim_with_no_surviving_evidence(mover, context, articles):
    bad = AnalystResult(
        headline="Nvidia announces something",
        narrative="Company news.",
        driver="company_news",
        confidence="high",
        evidence=[EvidenceItem(claim="Nvidia fell 4.5%", source_idx=0)],  # restatement, dropped
    )
    result = analyze(FakeLLM(AnalystResult=bad), mover, context, articles)
    assert result.evidence == []
    assert result.confidence == "low"


def test_struck_evidence_also_removes_the_prose_asserting_it(mover, context, articles):
    """The chips and the headline must not contradict each other: if the cause was
    struck, the narrative cannot go on asserting it."""
    bad = AnalystResult(
        headline="Amazon rises as AI price target lifted",
        narrative="The stock surged due to an analyst lifting the AI price target.",
        driver="analyst_action",
        confidence="high",
        evidence=[EvidenceItem(claim="lifted target boosted shares by 4.0%", source_idx=0)],
    )
    result = analyze(FakeLLM(AnalystResult=bad), mover, context, articles)

    assert result.evidence == []
    assert "price target" not in result.headline
    assert "price target" not in result.narrative
    assert "no supported catalyst" in result.headline


def test_data_backed_driver_without_citations_is_not_high_confidence(mover, context, articles):
    """Sector attribution rests on price arithmetic, which justifies medium, not high."""
    overconfident = AnalystResult(
        headline="Nvidia falls with the sector",
        narrative="Chips fell together.",
        driver="sector",
        confidence="high",
        evidence=[],
    )
    result = analyze(FakeLLM(AnalystResult=overconfident), mover, context, articles)
    assert result.driver == "sector"
    assert result.confidence == "medium"


def test_python_overrides_driver_for_sector_wide_moves(mover, context, articles):
    """Observed failure: the model labelled a sector selloff 'macro'."""
    wrong = AnalystResult(
        headline="Nvidia falls", narrative="n", driver="macro", confidence="medium"
    )
    result = analyze(FakeLLM(AnalystResult=wrong), mover, context, articles)
    assert result.driver == "sector"


# -- move classification ----------------------------------------------------------------


def test_classify_move_detects_sector_wide(mover, context):
    assert classify_move(mover, context) == "sector-wide"


def test_classify_move_detects_company_specific(mover, context):
    mover.sector_pct = 0.1
    mover.residual_pct = -4.5
    assert classify_move(mover, context) == "company-specific"


def test_move_brief_contains_no_model_numbers(mover, context):
    brief = move_brief(mover, context)
    assert "-4.57%" in brief and "beta of 2.07" in brief
    assert "ASSESSMENT" in brief


# -- triage -----------------------------------------------------------------------------


def test_triage_discards_invented_indices(mover, context, articles):
    client = FakeLLM(
        TriageResult=TriageResult(
            selected=[
                TriageSelection(idx=0, relevance="high", why="ok"),
                TriageSelection(idx=42, relevance="high", why="invented"),
            ]
        )
    )
    chosen = triage(client, mover, context, articles, max_articles=4)
    assert chosen == [articles[0]]


def test_triage_honours_no_material_news(mover, context, articles):
    client = FakeLLM(TriageResult=TriageResult(selected=[], no_material_news=True))
    assert triage(client, mover, context, articles, max_articles=4) == []


def test_triage_falls_back_to_recency_when_model_fails(mover, context, articles):
    client = FakeLLM(TriageResult=None)
    assert triage(client, mover, context, articles, max_articles=4) == articles


def test_triage_orders_by_relevance(mover, context, articles):
    client = FakeLLM(
        TriageResult=TriageResult(
            selected=[
                TriageSelection(idx=1, relevance="low", why=""),
                TriageSelection(idx=0, relevance="high", why=""),
            ]
        )
    )
    assert triage(client, mover, context, articles, max_articles=4) == [articles[0], articles[1]]


# -- verifier ---------------------------------------------------------------------------


def test_verifier_strikes_claims_and_downgrades_confidence(mover, articles):
    finding = Finding(
        mover=mover,
        articles=articles,
        analysis=AnalystResult(
            headline="h",
            narrative="n",
            driver="company_news",
            confidence="high",
            evidence=[
                EvidenceItem(claim="supported", source_idx=0),
                EvidenceItem(claim="invented", source_idx=1),
            ],
        ),
    )
    client = FakeLLM(
        VerifierResult=VerifierResult(
            drops=[VerifierDrop(symbol="NVDA", claim_idx=1, reason="not stated")]
        )
    )
    verify(client, [finding])
    assert [e.claim for e in finding.analysis.evidence] == ["supported"]
    assert finding.dropped_claims == ["invented"]
    assert finding.analysis.confidence == "medium"


def test_verifier_failure_leaves_claims_intact(mover, articles):
    """Verification is a safety net, not a gate - its failure must not empty the report."""
    finding = Finding(
        mover=mover,
        articles=articles,
        analysis=AnalystResult(
            headline="h", narrative="n", driver="sector", confidence="medium",
            evidence=[EvidenceItem(claim="kept", source_idx=0)],
        ),
    )
    verify(FakeLLM(VerifierResult=None), [finding])
    assert len(finding.analysis.evidence) == 1


def test_verifier_skips_degraded_findings(mover):
    finding = Finding(mover=mover, degraded=True)
    client = FakeLLM()
    verify(client, [finding])
    assert client.calls == 0, "nothing to check means no LLM call"


def test_downgrade_floors_at_low():
    assert downgrade("high") == "medium"
    assert downgrade("medium") == "low"
    assert downgrade("low") == "low"


# -- cited articles ---------------------------------------------------------------------


def test_cited_articles_pairs_claims_with_sources(mover, articles):
    finding = Finding(
        mover=mover,
        articles=articles,
        analysis=AnalystResult(
            headline="h", narrative="n", driver="sector", confidence="low",
            evidence=[EvidenceItem(claim="c", source_idx=1)],
        ),
    )
    pairs = finding.cited_articles
    assert pairs == [("c", articles[1])]
