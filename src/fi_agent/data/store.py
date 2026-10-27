"""SQLite persistence: article text cache, price snapshots, run history.

Stdlib sqlite3, no ORM. Three jobs:

* cache extracted article text so reruns cost no network,
* remember which article URLs have already been cited, so a story is not re-reported
  on every run of a watch loop,
* keep price snapshots so a run can show movement since the previous run.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from fi_agent.schemas import Quote

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    url_hash    TEXT PRIMARY KEY,
    url         TEXT NOT NULL,
    text        TEXT NOT NULL,
    fetched_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cited (
    url_hash    TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    run_id      TEXT NOT NULL,
    cited_at    TEXT NOT NULL,
    PRIMARY KEY (url_hash, symbol)
);
CREATE TABLE IF NOT EXISTS snapshots (
    run_id      TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    ts          TEXT NOT NULL,
    price       REAL NOT NULL,
    prev_close  REAL NOT NULL,
    pct_change  REAL NOT NULL,
    volume      REAL,
    PRIMARY KEY (run_id, symbol)
);
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    n_movers    INTEGER DEFAULT 0,
    status      TEXT DEFAULT 'running',
    report_dir  TEXT
);
CREATE TABLE IF NOT EXISTS flagged (
    run_id      TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    flagged_at  TEXT NOT NULL,
    PRIMARY KEY (run_id, symbol)
);
CREATE INDEX IF NOT EXISTS idx_snapshots_symbol ON snapshots(symbol, ts);
CREATE INDEX IF NOT EXISTS idx_flagged_run ON flagged(run_id);
"""


def _hash(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]


def _now() -> str:
    return datetime.now(UTC).isoformat()


class Store:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with closing(self.conn.cursor()) as cur:
            cur.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- article text cache ------------------------------------------------------------

    def get_article_text(self, url: str) -> str | None:
        row = self.conn.execute(
            "SELECT text FROM articles WHERE url_hash = ?", (_hash(url),)
        ).fetchone()
        return row["text"] if row else None

    def put_article_text(self, url: str, text: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO articles (url_hash, url, text, fetched_at) VALUES (?,?,?,?)",
            (_hash(url), url, text, _now()),
        )
        self.conn.commit()

    # -- cross-run citation memory -----------------------------------------------------

    def was_cited(self, url: str, symbol: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM cited WHERE url_hash = ? AND symbol = ?", (_hash(url), symbol)
        ).fetchone()
        return row is not None

    def mark_cited(self, url: str, symbol: str, run_id: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO cited (url_hash, symbol, run_id, cited_at) VALUES (?,?,?,?)",
            (_hash(url), symbol, run_id, _now()),
        )
        self.conn.commit()

    # -- price snapshots ---------------------------------------------------------------

    def save_snapshot(self, run_id: str, quote: Quote) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO snapshots
               (run_id, symbol, ts, price, prev_close, pct_change, volume)
               VALUES (?,?,?,?,?,?,?)""",
            (
                run_id,
                quote.symbol,
                _now(),
                quote.price,
                quote.prev_close,
                quote.pct_change,
                quote.volume,
            ),
        )
        self.conn.commit()

    def previous_price(self, symbol: str, before_run_id: str) -> float | None:
        """Price recorded for this symbol in the most recent earlier run."""
        row = self.conn.execute(
            """SELECT price FROM snapshots
               WHERE symbol = ? AND run_id != ?
               ORDER BY ts DESC LIMIT 1""",
            (symbol, before_run_id),
        ).fetchone()
        return float(row["price"]) if row else None

    # -- which tickers were flagged, per run -------------------------------------------

    def save_flagged(self, run_id: str, symbols: list[str]) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO flagged (run_id, symbol, flagged_at) VALUES (?,?,?)",
            [(run_id, symbol, _now()) for symbol in symbols],
        )
        self.conn.commit()

    def previously_flagged(self, before_run_id: str) -> set[str]:
        """Symbols flagged in the most recent earlier run.

        Used to email only about tickers that have *just* started moving. A name that
        stays flagged all afternoon should not generate a message every cycle.
        """
        row = self.conn.execute(
            """SELECT run_id FROM runs
               WHERE run_id != ? AND status = 'ok'
               ORDER BY started_at DESC LIMIT 1""",
            (before_run_id,),
        ).fetchone()
        if row is None:
            return set()
        rows = self.conn.execute(
            "SELECT symbol FROM flagged WHERE run_id = ?", (row["run_id"],)
        ).fetchall()
        return {r["symbol"] for r in rows}

    # -- run bookkeeping ---------------------------------------------------------------

    def start_run(self, run_id: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO runs (run_id, started_at, status) VALUES (?,?,'running')",
            (run_id, _now()),
        )
        self.conn.commit()

    def finish_run(self, run_id: str, n_movers: int, status: str, report_dir: str) -> None:
        self.conn.execute(
            """UPDATE runs SET finished_at = ?, n_movers = ?, status = ?, report_dir = ?
               WHERE run_id = ?""",
            (_now(), n_movers, status, report_dir, run_id),
        )
        self.conn.commit()

    def recent_runs(self, limit: int = 20) -> list[dict[str, object]]:
        rows = self.conn.execute(
            "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]

    def find_run(self, run_id: str) -> dict[str, object] | None:
        row = self.conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row else None


def dump_json(path: Path, payload: object) -> None:
    """Write run state for later replay, tolerating datetimes and Paths."""
    path.write_text(json.dumps(payload, indent=2, default=str))
