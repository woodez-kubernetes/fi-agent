"""Notion publishing: id parsing, block construction and the one-row-per-day lifecycle.

Every test here talks to an in-memory fake of the Notion API through httpx's
MockTransport. Nothing in this file contacts Notion.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest

from fi_agent import notion
from fi_agent.config import NotionSettings
from fi_agent.notion import (
    CHILDREN_LIMIT,
    TEXT_LIMIT,
    Credentials,
    build_blocks,
    database_id_from,
    make_client,
    maybe_publish,
    publish,
    text,
    trading_day,
)
from fi_agent.report.render import Diagnostics, build_view
from fi_agent.schemas import AnalystResult, EvidenceItem, ExecutiveSummary, Finding

DB = "3f6808aa11112222333344445555ffff"
TOKEN = "ntn_secret_token_value"

SCHEMA = {
    "Doc name": {"type": "title"},
    "Report date": {"type": "date"},
    "Model": {"type": "rich_text"},
    "Category": {"type": "multi_select"},
    "Ticker": {"type": "select"},
}


class FakeNotion:
    """Just enough of the Notion API to create, find, clear and append to pages."""

    def __init__(self, schema: dict | None = None) -> None:
        self.schema = schema or SCHEMA
        self.pages: dict[str, dict] = {}
        self.children: dict[str, list[dict]] = {}
        self.calls: list[tuple[str, str, dict | None]] = []
        self.fail_with: int | None = None
        self.rate_limit_once = False
        self._next = 0

    def _id(self) -> str:
        self._next += 1
        return f"id{self._next}"

    def _add_children(self, parent: str, blocks: list[dict]) -> None:
        for block in blocks:
            self.children.setdefault(parent, []).append({**block, "id": self._id()})

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        path = request.url.path.removeprefix("/v1")
        self.calls.append((request.method, path, body))

        if self.rate_limit_once:
            self.rate_limit_once = False
            return httpx.Response(429, headers={"Retry-After": "0"}, json={"code": "rate_limited"})
        if self.fail_with:
            return httpx.Response(
                self.fail_with, json={"code": "unauthorized", "message": "bad token"}
            )

        if request.method == "GET" and path == f"/databases/{DB}":
            return httpx.Response(
                200, json={"title": [{"plain_text": "Reports"}], "properties": self.schema}
            )
        if request.method == "POST" and path == f"/databases/{DB}/query":
            wanted = body["filter"]["title"]["equals"]
            hits = [p for p in self.pages.values() if p["title"] == wanted]
            return httpx.Response(200, json={"results": hits[:1]})
        if request.method == "POST" and path == "/pages":
            page_id = self._id()
            title = body["properties"]["Doc name"]["title"][0]["text"]["content"]
            self.pages[page_id] = {
                "id": page_id,
                "url": f"https://notion.so/{page_id}",
                "title": title,
                "properties": body["properties"],
            }
            self._add_children(page_id, body.get("children", []))
            return httpx.Response(200, json=self.pages[page_id])
        if request.method == "PATCH" and path.startswith("/pages/"):
            self.pages[path.split("/")[2]]["properties"] = body["properties"]
            return httpx.Response(200, json={})
        if path.startswith("/blocks/") and path.endswith("/children"):
            parent = path.split("/")[2]
            if request.method == "GET":
                items = self.children.get(parent, [])
                return httpx.Response(
                    200, json={"results": items, "has_more": False, "next_cursor": None}
                )
            self._add_children(parent, body["children"])
            return httpx.Response(200, json={})
        if request.method == "DELETE" and path.startswith("/blocks/"):
            block_id = path.split("/")[2]
            for blocks in self.children.values():
                blocks[:] = [b for b in blocks if b["id"] != block_id]
            return httpx.Response(200, json={})
        return httpx.Response(404, json={"code": "object_not_found", "message": path})

    def client(self) -> httpx.Client:
        return make_client(TOKEN, transport=httpx.MockTransport(self.handler))


@pytest.fixture
def fake() -> FakeNotion:
    return FakeNotion()


@pytest.fixture
def credentials() -> Credentials:
    return Credentials(TOKEN, DB)


@pytest.fixture
def finding(mover, articles) -> Finding:
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


@pytest.fixture
def state(finding, context, quote_factory, mover) -> dict:
    quiet = mover.model_copy(
        update={"symbol": "AAPL", "name": "Apple", "quote": quote_factory("AAPL", 201, 200)}
    )
    return {
        "run_id": "2026-10-09_183000",
        # 22:30 ET on the 9th: still the 9th in New York, already the 10th in UTC.
        "generated_at": datetime(2026, 10, 10, 2, 30, tzinfo=UTC),
        "context": context,
        "summary": ExecutiveSummary(summary="Chips fell.", themes=["semis"]),
        "findings": [finding],
        "quiet": [quiet],
        "errors": [],
        "diagnostics": Diagnostics(
            model="qwen3:8b", base_url="x", llm_calls=4, llm_seconds=12.0,
            duration_s=30.0, degraded=[],
        ),
    }


def _types(blocks: list[dict]) -> list[str]:
    return [b["type"] for b in blocks]


def _plain(rich_text: list[dict]) -> str:
    return "".join(r["text"]["content"] for r in rich_text)


# --------------------------------------------------------------------------------------
# Parsing and text
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        f"https://www.notion.so/woodcap/Reports-{DB}?v=99998888777766665555444433332222",
        f"https://www.notion.so/{DB}",
        f"https://www.notion.so/woodcap/Code-{DB}",  # title ends in hex-like letters
        DB,
        f"{DB[:8]}-{DB[8:12]}-{DB[12:16]}-{DB[16:20]}-{DB[20:]}",
    ],
)
def test_database_id_from_accepts_urls_and_ids(value):
    assert database_id_from(value) == DB


def test_database_id_from_rejects_garbage():
    assert database_id_from("https://www.notion.so/not-a-page") is None


def test_text_splits_long_content_and_drops_bad_links():
    runs = text("x" * (TEXT_LIMIT + 5), link="javascript:alert(1)")
    assert [len(r["text"]["content"]) for r in runs] == [TEXT_LIMIT, 5]
    assert all(r["text"]["link"] is None for r in runs)
    assert text("") == []


def test_trading_day_uses_new_york_time(state):
    assert trading_day(state["generated_at"]).isoformat() == "2026-10-09"


# --------------------------------------------------------------------------------------
# Blocks
# --------------------------------------------------------------------------------------


def test_build_blocks_layout(state):
    blocks = build_blocks(build_view(**state))
    types = _types(blocks)
    assert types[0] == "callout"
    assert types.count("table") == 2  # movers and the rest of the watchlist
    assert "quote" in types

    movers_table = next(b for b in blocks if b["type"] == "table")
    assert movers_table["table"]["table_width"] == 7
    assert len(movers_table["table"]["children"]) == 2  # header + NVDA
    change_cell = movers_table["table"]["children"][1]["table_row"]["cells"][2]
    assert change_cell[0]["annotations"]["color"] == "red"

    toggle = next(b for b in blocks if b["type"] == "heading_3")
    assert toggle["heading_3"]["is_toggleable"] is True
    inner = toggle["heading_3"]["children"]
    assert "bulleted_list_item" in _types(inner)
    citation = inner[_types(inner).index("bulleted_list_item")]["bulleted_list_item"]
    assert any(r["text"]["link"] == {"url": "https://example.com/a"} for r in citation["rich_text"])


def test_build_blocks_without_findings(state):
    state["findings"] = []
    blocks = build_blocks(build_view(**state))
    assert "heading_3" not in _types(blocks)
    assert any(
        "No watchlist name cleared" in _plain(b["paragraph"]["rich_text"])
        for b in blocks
        if b["type"] == "paragraph"
    )


def test_build_blocks_degraded_finding(state, mover, articles):
    state["findings"] = [
        Finding(mover=mover, articles=articles, degraded=True, degraded_reason="LLM down")
    ]
    blocks = build_blocks(build_view(**state))
    toggle = next(b for b in blocks if b["type"] == "heading_3")
    first = toggle["heading_3"]["children"][0]["paragraph"]["rich_text"]
    assert "LLM down" in _plain(first)


def test_errors_get_a_warning_callout(state):
    state["errors"] = ["feed timed out"]
    blocks = build_blocks(build_view(**state))
    assert blocks[1]["type"] == "callout"
    assert blocks[1]["callout"]["color"] == "red_background"


# --------------------------------------------------------------------------------------
# Page lifecycle
# --------------------------------------------------------------------------------------


def test_first_run_of_the_day_creates_a_row(fake, credentials, state):
    result = publish(NotionSettings(enabled=True), credentials, state, client=fake.client())

    assert result.published, result.detail
    assert result.url and "notion.so" in result.url
    (page,) = fake.pages.values()
    assert page["title"] == "Watchlist · 2026-10-09"
    props = page["properties"]
    assert props["Report date"] == {"date": {"start": "2026-10-09"}}
    assert _plain(props["Model"]["rich_text"]) == "qwen3:8b"
    assert props["Category"] == {"multi_select": [{"name": "Watchlist"}]}
    assert "Ticker" not in props  # unrelated columns are left alone
    assert fake.children[page["id"]]


def test_later_run_same_day_replaces_the_body(fake, credentials, state):
    settings = NotionSettings(enabled=True)
    publish(settings, credentials, state, client=fake.client())
    (page_id,) = fake.pages
    first_ids = {b["id"] for b in fake.children[page_id]}

    state["summary"] = ExecutiveSummary(summary="Chips recovered.")
    result = publish(settings, credentials, state, client=fake.client())

    assert result.published and result.detail.startswith("updated")
    assert len(fake.pages) == 1  # no second row
    body = fake.children[page_id]
    assert not first_ids & {b["id"] for b in body}
    quotes = [b for b in body if b["type"] == "quote"]
    assert _plain(quotes[0]["quote"]["rich_text"]) == "Chips recovered."


def test_next_trading_day_gets_a_new_row(fake, credentials, state):
    settings = NotionSettings(enabled=True)
    publish(settings, credentials, state, client=fake.client())
    state["generated_at"] = datetime(2026, 10, 10, 15, 0, tzinfo=UTC)
    publish(settings, credentials, state, client=fake.client())
    assert sorted(p["title"] for p in fake.pages.values()) == [
        "Watchlist · 2026-10-09",
        "Watchlist · 2026-10-10",
    ]


def test_large_reports_are_sent_in_chunks(fake, credentials, state, finding):
    state["findings"] = [finding] * (CHILDREN_LIMIT + 20)
    result = publish(NotionSettings(enabled=True), credentials, state, client=fake.client())

    assert result.published
    create = next(c for c in fake.calls if c[:2] == ("POST", "/pages"))
    assert len(create[2]["children"]) == CHILDREN_LIMIT
    appends = [c for c in fake.calls if c[0] == "PATCH" and c[1].endswith("/children")]
    assert appends and all(len(c[2]["children"]) <= CHILDREN_LIMIT for c in appends)


def test_missing_columns_are_skipped(credentials, state):
    fake = FakeNotion(schema={"Doc name": {"type": "title"}, "Model": {"type": "number"}})
    publish(NotionSettings(enabled=True), credentials, state, client=fake.client())
    (page,) = fake.pages.values()
    assert set(page["properties"]) == {"Doc name"}


def test_rate_limit_is_retried(fake, credentials, state, monkeypatch):
    monkeypatch.setattr(notion.time, "sleep", lambda _s: None)
    fake.rate_limit_once = True
    result = publish(NotionSettings(enabled=True), credentials, state, client=fake.client())
    assert result.published


def test_api_errors_are_reported_not_raised(fake, credentials, state):
    fake.fail_with = 401
    result = publish(NotionSettings(enabled=True), credentials, state, client=fake.client())
    assert not result.published
    assert "401" in result.detail
    assert TOKEN not in result.detail


# --------------------------------------------------------------------------------------
# Gating and credentials
# --------------------------------------------------------------------------------------


@pytest.fixture
def no_dotenv(monkeypatch):
    """Stop load_credentials reading the real .env at the project root."""
    monkeypatch.setattr(notion, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv(notion.TOKEN_ENV, raising=False)
    monkeypatch.delenv(notion.DATABASE_ENV, raising=False)


def test_disabled_does_nothing(state, no_dotenv):
    assert maybe_publish(NotionSettings(enabled=False), state, ["NVDA"]).detail == (
        "notion disabled"
    )


def test_nothing_flagged_is_skipped(state, no_dotenv):
    result = maybe_publish(NotionSettings(enabled=True), state, [])
    assert not result.published and result.detail == "nothing flagged"


def test_missing_credentials_are_reported(state, no_dotenv):
    result = maybe_publish(NotionSettings(enabled=True), state, ["NVDA"])
    assert not result.published and notion.TOKEN_ENV in result.detail


def test_load_credentials_parses_database_url(no_dotenv, monkeypatch):
    monkeypatch.setenv(notion.TOKEN_ENV, TOKEN)
    monkeypatch.setenv(notion.DATABASE_ENV, f"https://www.notion.so/x/Reports-{DB}?v=1234")
    credentials, problem = notion.load_credentials()
    assert credentials == Credentials(TOKEN, DB) and problem == ""


def test_check_explains_unshared_database(credentials):
    fake = FakeNotion()
    other = Credentials(TOKEN, "0" * 32)
    ok, detail = notion.check(other, client=fake.client())
    assert not ok and "share it" in detail
    assert notion.check(credentials, client=fake.client()) == (True, "database 'Reports' reachable")
