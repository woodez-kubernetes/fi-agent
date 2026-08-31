"""Rendering, sparklines, persistence and the replay round-trip."""

from __future__ import annotations

import json

from fi_agent.data.store import Store
from fi_agent.publish import publish, rerender, run_id_for
from fi_agent.report.render import Diagnostics, build_view, render_html, render_markdown
from fi_agent.report.sparkline import sparkline
from fi_agent.schemas import AnalystResult, EvidenceItem, ExecutiveSummary, Finding

from .conftest import make_quote


def diagnostics() -> Diagnostics:
    return Diagnostics(
        model="qwen2.5:7b", base_url="http://test:11434", llm_calls=8,
        llm_seconds=159.9, duration_s=74.7, degraded=[],
    )


def make_finding(mover, articles) -> Finding:
    return Finding(
        mover=mover,
        articles=articles,
        analysis=AnalystResult(
            headline="Nvidia slides with the chip sector",
            narrative="Peers fell together after a supply warning.",
            driver="sector",
            confidence="medium",
            evidence=[EvidenceItem(claim="SK Hynix warned on memory supply", source_idx=0)],
            watch_next="Whether peers recover.",
        ),
    )


# -- sparkline --------------------------------------------------------------------------


def test_sparkline_renders_svg():
    svg = sparkline([1.0, 2.0, 1.5, 3.0])
    assert svg.startswith("<svg") and "polyline" in svg


def test_sparkline_needs_two_points():
    assert sparkline([]) == ""
    assert sparkline([1.0]) == ""


def test_sparkline_handles_flat_series_without_dividing_by_zero():
    svg = sparkline([5.0, 5.0, 5.0])
    assert "<svg" in svg and "nan" not in svg.lower()


def test_sparkline_colour_follows_direction():
    assert "--up" in sparkline([1.0, 2.0])
    assert "--down" in sparkline([2.0, 1.0])


def test_sparkline_escapes_label():
    assert "<script>" not in sparkline([1.0, 2.0], label="<script>x</script>")


# -- rendering --------------------------------------------------------------------------


def test_html_report_is_self_contained(mover, articles, context):
    view = build_view(
        "run1", [make_finding(mover, articles)], [], context,
        ExecutiveSummary(summary="A quiet session.", themes=["chips"], top_story="NVDA"),
        [], diagnostics(),
    )
    html = render_html(view)
    assert "<!doctype html>" in html.lower()
    assert "{{" not in html and "{%" not in html
    assert 'src="http' not in html and 'link rel="stylesheet"' not in html
    assert "not\n    investment advice" in html or "investment advice" in html


def test_html_shows_computed_numbers_not_model_text(mover, articles, context):
    view = build_view(
        "run1", [make_finding(mover, articles)], [], context,
        ExecutiveSummary(summary="s"), [], diagnostics(),
    )
    html = render_html(view)
    assert "-4.57%" in html, "percentage comes from the Quote, not the model"
    assert "217.55" in html


def test_html_escapes_model_output(mover, articles, context):
    finding = make_finding(mover, articles)
    finding.analysis.narrative = "<script>alert(1)</script>"
    view = build_view(
        "run1", [finding], [], context, ExecutiveSummary(summary="s"), [], diagnostics()
    )
    assert "<script>alert(1)</script>" not in render_html(view)


def test_degraded_finding_renders_without_analysis(mover, articles, context):
    finding = Finding(
        mover=mover, articles=articles, degraded=True, degraded_reason="model timed out"
    )
    view = build_view(
        "run1", [finding], [], context, ExecutiveSummary(summary="s"), [], diagnostics()
    )
    html = render_html(view)
    assert "model timed out" in html
    assert "-4.57%" in html, "a degraded ticker still shows its market data"


def test_empty_run_renders(context):
    view = build_view(
        "run1", [], [], context, ExecutiveSummary(summary="Nothing moved."), [], diagnostics()
    )
    html = render_html(view)
    assert "No watchlist name cleared" in html


def test_markdown_separates_ticker_sections(mover, articles, context):
    view = build_view(
        "run1", [make_finding(mover, articles)], [], context,
        ExecutiveSummary(summary="s"), [], diagnostics(),
    )
    md = render_markdown(view)
    # Regression: triggers used to run straight into the next heading.
    assert "\n### NVDA" in md
    for line in md.splitlines():
        assert not (line.startswith("Triggers:") and "###" in line)


def test_findings_ordered_by_idiosyncratic_move(context, articles):
    from fi_agent.schemas import Mover

    small = Finding(
        mover=Mover(symbol="A", name="A", quote=make_quote("A", 101.0, 100.0), residual_pct=1.0)
    )
    large = Finding(
        mover=Mover(symbol="B", name="B", quote=make_quote("B", 102.0, 100.0), residual_pct=-5.0)
    )
    view = build_view(
        "r", [small, large], [], context, ExecutiveSummary(summary="s"), [], diagnostics()
    )
    assert [f.mover.symbol for f in view["findings"]] == ["B", "A"]


# -- publish / replay -------------------------------------------------------------------


def test_publish_writes_all_artefacts_and_replays(tmp_path, mover, articles, context):
    reports = tmp_path / "reports"
    run_id = run_id_for()
    paths = publish(
        reports_dir=reports,
        run_id=run_id,
        findings=[make_finding(mover, articles)],
        quiet=[],
        context=context,
        summary=ExecutiveSummary(summary="A session.", themes=["chips"], top_story="NVDA"),
        errors=[],
        diagnostics=diagnostics(),
    )
    for key in ("html", "markdown", "state"):
        assert paths[key].exists()
    assert (reports / "latest.html").exists()

    original = paths["html"].read_text()
    # Replay must reproduce the report from stored state, with no LLM involved.
    rerender(paths["state"])
    assert paths["html"].read_text() == original


def test_state_json_is_valid_and_complete(tmp_path, mover, articles, context):
    paths = publish(
        reports_dir=tmp_path / "reports",
        run_id="r1",
        findings=[make_finding(mover, articles)],
        quiet=[],
        context=context,
        summary=ExecutiveSummary(summary="s"),
        errors=["something failed"],
        diagnostics=diagnostics(),
    )
    state = json.loads(paths["state"].read_text())
    assert state["run_id"] == "r1"
    assert state["errors"] == ["something failed"]
    assert state["findings"][0]["mover"]["symbol"] == "NVDA"


# -- store ------------------------------------------------------------------------------


def test_store_caches_article_text(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        assert store.get_article_text("https://x/1") is None
        store.put_article_text("https://x/1", "body text")
        assert store.get_article_text("https://x/1") == "body text"


def test_store_remembers_citations_across_runs(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        assert not store.was_cited("https://x/1", "NVDA")
        store.mark_cited("https://x/1", "NVDA", "run1")
        assert store.was_cited("https://x/1", "NVDA")
        assert not store.was_cited("https://x/1", "AAPL"), "citation memory is per ticker"


def test_store_tracks_previous_price(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        store.save_snapshot("run1", make_quote("NVDA", 200.0, 195.0))
        store.save_snapshot("run2", make_quote("NVDA", 210.0, 195.0))
        assert store.previous_price("NVDA", before_run_id="run2") == 200.0


def test_store_records_run_lifecycle(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        store.start_run("run1")
        store.finish_run("run1", 3, "ok", "/tmp/run1")
        run = store.find_run("run1")
        assert run["status"] == "ok" and run["n_movers"] == 3
