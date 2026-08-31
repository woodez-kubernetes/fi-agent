"""Write the report to disk and point `latest` at it.

Each run gets its own directory holding the rendered report, the raw state that produced
it, and the LLM trace. Keeping `state.json` is what makes `fi-agent replay` possible:
the template can be reworked and re-rendered without spending minutes of inference to
regenerate content that has not changed.
"""

from __future__ import annotations

import json
import logging
import webbrowser
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fi_agent.report.render import Diagnostics, build_view, render_html, render_markdown
from fi_agent.schemas import ExecutiveSummary, Finding, MarketContext, Mover

log = logging.getLogger(__name__)


def run_id_for(now: datetime | None = None) -> str:
    return (now or datetime.now(UTC)).strftime("%Y-%m-%d_%H%M%S")


def run_directory(reports_dir: Path, run_id: str) -> Path:
    path = reports_dir / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def serialise_state(
    run_id: str,
    findings: list[Finding],
    quiet: list[Mover],
    context: MarketContext,
    summary: ExecutiveSummary,
    errors: list[str],
    diagnostics: Diagnostics,
    generated_at: datetime,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "generated_at": generated_at.isoformat(),
        "context": context.model_dump(mode="json"),
        "summary": summary.model_dump(mode="json"),
        "findings": [f.model_dump(mode="json") for f in findings],
        "quiet": [m.model_dump(mode="json") for m in quiet],
        "errors": errors,
        "diagnostics": asdict(diagnostics),
    }


def load_state(path: Path) -> dict[str, Any]:
    """Rehydrate a saved run into the objects the renderer expects."""
    raw = json.loads(path.read_text())
    return {
        "run_id": raw["run_id"],
        "generated_at": datetime.fromisoformat(raw["generated_at"]),
        "context": MarketContext.model_validate(raw["context"]),
        "summary": ExecutiveSummary.model_validate(raw["summary"]),
        "findings": [Finding.model_validate(f) for f in raw["findings"]],
        "quiet": [Mover.model_validate(m) for m in raw["quiet"]],
        "errors": raw.get("errors", []),
        "diagnostics": Diagnostics(**raw["diagnostics"]),
    }


def _update_latest(reports_dir: Path, target: Path) -> None:
    """Point reports/latest.html at this run, falling back to a copy where symlinks
    are unavailable."""
    link = reports_dir / "latest.html"
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target.relative_to(reports_dir))
    except OSError as exc:
        log.debug("symlink failed (%s), copying instead", exc)
        try:
            link.write_text(target.read_text())
        except OSError as copy_exc:
            log.warning("could not update latest.html: %s", copy_exc)


def publish(
    reports_dir: Path,
    run_id: str,
    findings: list[Finding],
    quiet: list[Mover],
    context: MarketContext,
    summary: ExecutiveSummary,
    errors: list[str],
    diagnostics: Diagnostics,
    generated_at: datetime | None = None,
    open_browser: bool = False,
) -> dict[str, Path]:
    stamp = generated_at or datetime.now(UTC)
    directory = run_directory(reports_dir, run_id)

    view = build_view(
        run_id, findings, quiet, context, summary, errors, diagnostics, stamp
    )

    html_path = directory / "report.html"
    md_path = directory / "report.md"
    state_path = directory / "state.json"

    html_path.write_text(render_html(view), encoding="utf-8")
    md_path.write_text(render_markdown(view), encoding="utf-8")
    state_path.write_text(
        json.dumps(
            serialise_state(
                run_id, findings, quiet, context, summary, errors, diagnostics, stamp
            ),
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    _update_latest(reports_dir, html_path)

    if open_browser:
        webbrowser.open(html_path.resolve().as_uri())

    log.info("report written to %s", html_path)
    return {"html": html_path, "markdown": md_path, "state": state_path, "dir": directory}


def rerender(state_path: Path, open_browser: bool = False) -> dict[str, Path]:
    """Re-render a stored run with the current templates. No LLM calls."""
    state = load_state(state_path)
    directory = state_path.parent
    view = build_view(**state)

    html_path = directory / "report.html"
    md_path = directory / "report.md"
    html_path.write_text(render_html(view), encoding="utf-8")
    md_path.write_text(render_markdown(view), encoding="utf-8")
    _update_latest(directory.parent, html_path)

    if open_browser:
        webbrowser.open(html_path.resolve().as_uri())
    return {"html": html_path, "markdown": md_path, "dir": directory}
