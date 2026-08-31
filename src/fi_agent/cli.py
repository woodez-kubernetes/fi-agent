"""Command line interface."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from fi_agent import publish as publish_mod
from fi_agent.agents.graph import Deps, run_pipeline
from fi_agent.config import (
    TickerConfig,
    load_settings,
    load_watchlist,
    save_watchlist,
)
from fi_agent.data.market import market_is_open
from fi_agent.data.store import Store
from fi_agent.llm import LLMClient
from fi_agent.report.render import Diagnostics
from fi_agent.schemas import MarketContext

app = typer.Typer(
    add_completion=False,
    help="Multi-agent stock watchlist monitor running on a local Qwen LLM.",
    no_args_is_help=True,
)
watchlist_app = typer.Typer(help="Inspect and edit the watchlist.", no_args_is_help=True)
app.add_typer(watchlist_app, name="watchlist")

console = Console()


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    # These libraries are extremely chatty at DEBUG and drown out our own logs.
    for noisy in ("httpx", "httpcore", "urllib3", "yfinance", "peewee", "trafilatura"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@app.command()
def run(
    tickers: Annotated[
        str | None,
        typer.Option("--tickers", "-t", help="Comma-separated subset to run, e.g. NVDA,AMD"),
    ] = None,
    no_llm: Annotated[
        bool, typer.Option("--no-llm", help="Data and screening only; skip all agents")
    ] = False,
    open_browser: Annotated[
        bool, typer.Option("--open", help="Open the report when it is written")
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Fetch, screen, investigate and publish a watchlist report."""
    _configure_logging(verbose)
    settings = load_settings()
    watchlist = load_watchlist()

    if tickers:
        wanted = {t.strip().upper() for t in tickers.split(",") if t.strip()}
        watchlist.tickers = [t for t in watchlist.tickers if t.symbol.upper() in wanted]
        missing = wanted - {t.symbol.upper() for t in watchlist.tickers}
        if missing:
            # Run them anyway - a one-off look at a ticker should not require editing config.
            watchlist.tickers.extend(TickerConfig(symbol=s, name=s) for s in sorted(missing))

    if not watchlist.tickers:
        console.print("[red]Watchlist is empty.[/] Add tickers to config/watchlist.yaml.")
        raise typer.Exit(1)

    client: LLMClient | None = None
    run_id = publish_mod.run_id_for()
    reports_dir = settings.reports_path
    directory = publish_mod.run_directory(reports_dir, run_id)

    if not no_llm:
        client = LLMClient(settings.llm, trace_path=directory / "llm_trace.jsonl")
        ok, message = client.check()
        if not ok:
            console.print(f"[yellow]LLM unavailable:[/] {message}")
            console.print("[yellow]Continuing without narrative analysis.[/]")
            client = None

    started = time.monotonic()
    generated_at = datetime.now(UTC)

    with Store(settings.database_path) as store:
        store.start_run(run_id)
        deps = Deps(
            settings=settings,
            watchlist=watchlist,
            client=client,
            store=store,
            run_id=run_id,
        )
        with console.status("[bold]Collecting market data and investigating movers..."):
            state = asyncio.run(run_pipeline(deps))

        findings = state.get("findings", [])
        quiet = state.get("quiet", [])
        context = state.get("context") or MarketContext()
        errors = state.get("errors", [])

        diagnostics = Diagnostics(
            model=settings.llm.model if client else "none (data only)",
            base_url=settings.llm.base_url,
            llm_calls=client.calls if client else 0,
            llm_seconds=round(client.total_seconds, 1) if client else 0.0,
            duration_s=round(time.monotonic() - started, 1),
            degraded=[f.mover.symbol for f in findings if f.degraded],
        )

        paths = publish_mod.publish(
            reports_dir=reports_dir,
            run_id=run_id,
            findings=findings,
            quiet=quiet,
            context=context,
            summary=state["summary"],
            errors=errors,
            diagnostics=diagnostics,
            generated_at=generated_at,
            open_browser=open_browser,
        )
        store.finish_run(run_id, len(findings), "ok", str(paths["dir"]))

    _print_summary(findings, quiet, context, diagnostics, paths["html"])


def _print_summary(findings, quiet, context, diagnostics, html_path: Path) -> None:
    table = Table(title=None, box=None, pad_edge=False, header_style="dim")
    table.add_column("Ticker")
    table.add_column("Change", justify="right")
    table.add_column(f"vs {context.benchmark}", justify="right")
    table.add_column("Driver")

    for finding in sorted(
        findings, key=lambda f: abs(f.mover.quote.pct_change), reverse=True
    ):
        change = finding.mover.quote.pct_change
        residual = finding.mover.residual_pct
        table.add_row(
            finding.mover.symbol,
            f"[{'green' if change >= 0 else 'red'}]{change:+.2f}%[/]",
            f"{residual:+.2f}%" if residual is not None else "—",
            finding.analysis.driver.replace("_", " ") if finding.analysis else "no analysis",
        )

    console.print()
    if findings:
        console.print(table)
    else:
        console.print("[dim]No names cleared screening.[/]")
    console.print(
        f"\n{len(findings)} flagged, {len(quiet)} quiet · "
        f"{diagnostics.llm_calls} LLM calls in {diagnostics.llm_seconds}s · "
        f"{diagnostics.duration_s}s total"
    )
    console.print(f"[bold]Report:[/] {html_path}")


@app.command()
def replay(
    run: Annotated[str, typer.Argument(help="Run id, or 'latest'")] = "latest",
    open_browser: Annotated[bool, typer.Option("--open")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Re-render a stored run with the current templates. Makes no LLM calls."""
    _configure_logging(verbose)
    settings = load_settings()
    reports_dir = settings.reports_path

    if run == "latest":
        candidates = sorted(p for p in reports_dir.glob("*/state.json"))
        if not candidates:
            console.print("[red]No stored runs found.[/]")
            raise typer.Exit(1)
        state_path = candidates[-1]
    else:
        state_path = reports_dir / run / "state.json"
        if not state_path.exists():
            console.print(f"[red]No such run:[/] {run}")
            raise typer.Exit(1)

    paths = publish_mod.rerender(state_path, open_browser=open_browser)
    console.print(f"[bold]Re-rendered:[/] {paths['html']}")


@app.command()
def doctor(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    """Check that everything the agent depends on is reachable."""
    _configure_logging(verbose)
    import sys

    settings = load_settings()
    watchlist = load_watchlist()
    checks: list[tuple[str, bool, str]] = []

    version = ".".join(str(v) for v in sys.version_info[:3])
    in_venv = sys.prefix != sys.base_prefix
    checks.append(("Python 3.13", sys.version_info[:2] == (3, 13), version))
    checks.append(("Virtualenv active", in_venv, sys.prefix))

    ok, message = LLMClient(settings.llm).check()
    checks.append(("Ollama", ok, message))

    try:
        from fi_agent.data.market import fetch_quotes

        quotes, _, failed = fetch_quotes([settings.screening.benchmark])
        quote = quotes.get(settings.screening.benchmark)
        checks.append(
            (
                "Market data",
                quote is not None,
                f"{settings.screening.benchmark} {quote.price:.2f} ({quote.pct_change:+.2f}%)"
                if quote
                else f"failed: {failed}",
            )
        )
    except Exception as exc:
        checks.append(("Market data", False, str(exc)))

    try:
        import asyncio as _asyncio

        from fi_agent.data.news import gather_headlines

        first = watchlist.tickers[0] if watchlist.tickers else None
        if first:
            articles = _asyncio.run(
                gather_headlines(first.symbol, first.display_name(), settings.news)
            )
            checks.append(
                ("News feeds", bool(articles), f"{len(articles)} headlines for {first.symbol}")
            )
    except Exception as exc:
        checks.append(("News feeds", False, str(exc)))

    checks.append(
        ("Watchlist", bool(watchlist.tickers), f"{len(watchlist.tickers)} tickers")
    )
    checks.append(("Market session", True, "open" if market_is_open() else "closed"))

    table = Table(box=None, pad_edge=False, header_style="dim")
    table.add_column("Check")
    table.add_column("")
    table.add_column("Detail", overflow="fold")
    for name, passed, detail in checks:
        table.add_row(name, "[green]ok[/]" if passed else "[red]fail[/]", detail)
    console.print(table)

    if not all(passed for _, passed, _ in checks):
        raise typer.Exit(1)


@app.command()
def watch(
    interval: Annotated[
        int, typer.Option("--interval", "-i", help="Minutes between runs")
    ] = 30,
    market_hours_only: Annotated[
        bool, typer.Option("--market-hours/--always", help="Only run while the market is open")
    ] = True,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run repeatedly on an interval."""
    _configure_logging(verbose)
    console.print(
        f"Watching every {interval} min"
        f"{' during market hours' if market_hours_only else ''}. Ctrl-C to stop."
    )
    try:
        while True:
            if market_hours_only and not market_is_open():
                console.print("[dim]Market closed, skipping this cycle.[/]")
            else:
                try:
                    run(tickers=None, no_llm=False, open_browser=False, verbose=verbose)
                except Exception as exc:
                    console.print(f"[red]Run failed:[/] {exc}")
            time.sleep(interval * 60)
    except KeyboardInterrupt:
        console.print("\nStopped.")


@watchlist_app.command("list")
def watchlist_list() -> None:
    """Show the configured watchlist."""
    watchlist = load_watchlist()
    settings = load_settings()
    table = Table(box=None, pad_edge=False, header_style="dim")
    table.add_column("Symbol")
    table.add_column("Name")
    table.add_column("Sector ETF")
    table.add_column("Threshold", justify="right")
    for ticker in watchlist.tickers:
        table.add_row(
            ticker.symbol,
            ticker.name,
            ticker.sector_etf or "—",
            f"{ticker.move_threshold_pct or settings.screening.move_threshold_pct:.1f}%",
        )
    console.print(table)


@watchlist_app.command("add")
def watchlist_add(
    symbol: Annotated[str, typer.Argument(help="Ticker symbol")],
    name: Annotated[str, typer.Option("--name", "-n")] = "",
    sector_etf: Annotated[str | None, typer.Option("--sector-etf", "-s")] = None,
    threshold: Annotated[float | None, typer.Option("--threshold")] = None,
) -> None:
    """Add a ticker to the watchlist."""
    watchlist = load_watchlist()
    symbol = symbol.upper()
    if watchlist.get(symbol):
        console.print(f"[yellow]{symbol} is already on the watchlist.[/]")
        raise typer.Exit(1)
    watchlist.tickers.append(
        TickerConfig(
            symbol=symbol, name=name or symbol, sector_etf=sector_etf,
            move_threshold_pct=threshold,
        )
    )
    path = save_watchlist(watchlist)
    console.print(f"Added [bold]{symbol}[/] to {path}")


@watchlist_app.command("remove")
def watchlist_remove(symbol: Annotated[str, typer.Argument(help="Ticker symbol")]) -> None:
    """Remove a ticker from the watchlist."""
    watchlist = load_watchlist()
    symbol = symbol.upper()
    if not watchlist.get(symbol):
        console.print(f"[yellow]{symbol} is not on the watchlist.[/]")
        raise typer.Exit(1)
    watchlist.tickers = [t for t in watchlist.tickers if t.symbol != symbol]
    path = save_watchlist(watchlist)
    console.print(f"Removed [bold]{symbol}[/] from {path}")


if __name__ == "__main__":
    app()
