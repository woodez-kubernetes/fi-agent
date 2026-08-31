# fi-agent: multi-agent stock watchlist monitor

## Context

`fi-agent` is an empty repo (README, .gitignore, CLAUDE.md). The goal is a multi-agent system,
running entirely on a local Qwen LLM, that:

1. watches a configured stock watchlist,
2. detects meaningful price/volume moves,
3. hunts for news that explains each move,
4. writes a report attributing moves to causes, with citations,
5. publishes it as a readable HTML + Markdown document.

**Verified environment facts** (probed during planning, not assumed):

| Fact | Value |
|---|---|
| Ollama | v0.20.0 at `http://192.168.2.167:11434`, reachable |
| Models available | `qwen2.5:7b` only (Q4_K_M, 32k ctx) |
| Tool calling | ✅ works — returned a valid `get_quote` tool call |
| JSON-schema output (`format`) | ✅ works — returned schema-conforming JSON |
| Generation speed | ~20 tok/s; ~8s for a small call |
| **Parallel requests** | ⚠️ **effectively serialized** — 2 concurrent calls took 10.6s vs 12.1s sequential |
| Yahoo chart API | ✅ live quotes returned for NVDA |
| Yahoo Finance RSS | ✅ working |
| Google News RSS | ✅ working, fresh headlines |
| Local Python | `python3.13` at `/usr/local/bin/python3.13` (default `python3` is 3.14 — don't use it) |
| Finance API keys | none in environment |

**Decisions made** (user-selected): publish as HTML + Markdown to disk; free data via
yfinance + RSS; LangGraph custom supervisor; stay on `qwen2.5:7b`.

## Three design principles the whole plan hangs on

These follow directly from the probe results — a 4-bit 7B model that serializes requests.

1. **The LLM never touches a number.** Every price, percentage, volume ratio, and beta residual
   is computed in Python and injected into the template. The model only ranks headlines,
   attributes causes from prose, and writes summary text. This makes numeric hallucination
   structurally impossible rather than merely unlikely.

2. **Every LLM hop returns a JSON-schema-validated object**, using Ollama's `format` parameter
   plus a Pydantic model. **Enums, not floats** — the probe asked for a 0–1 `confidence` float
   and the model returned `100`. Use `"high"|"medium"|"low"` everywhere a score is wanted.

3. **Minimize LLM call count; parallelize only I/O.** Since the server serializes inference,
   wall-clock ≈ sum of LLM calls. Budget **2 calls per flagged mover + 3 fixed** (~10 calls,
   3–5 min for a typical run). Network work (RSS, article fetch) fans out async and is free.

## Architecture

```
load_config → fetch_market → screen → market_context
                                           │
                            Send() fan-out over flagged movers
                    ┌──────────────────────┼──────────────────────┐
              ticker subgraph        ticker subgraph        ticker subgraph
              1. gather_news  (async I/O, no LLM)
              2. triage_news  (1 LLM call)
              3. analyze      (1 LLM call)
                    └──────────────────────┼──────────────────────┘
                                           │  (reducer fan-in)
                                        verify      (1 LLM call, batched)
                                           ↓
                                       synthesize   (1 LLM call)
                                           ↓
                                        render      (Jinja2, no LLM)
                                           ↓
                                        publish     (disk + browser)
```

Nodes with no LLM call are plain deterministic Python — most of the system is ordinary code,
and the agents are confined to the three places judgment is genuinely required.

## Layout

```
fi-agent/
├── CLAUDE.md                 (exists)
├── requirements.txt          runtime + dev deps
├── config/
│   ├── watchlist.yaml        tickers, sector ETF, per-ticker threshold overrides
│   └── settings.yaml         llm url/model/num_ctx, thresholds, lookback, paths
├── src/fi_agent/
│   ├── config.py             pydantic-settings models, YAML + .env loading
│   ├── llm.py                Ollama client, structured_call(), retry, trace log
│   ├── schemas.py            all Pydantic I/O models (LLM contracts + domain)
│   ├── data/
│   │   ├── market.py         yfinance quotes, 60d history, volume stats
│   │   ├── news.py           Yahoo RSS + Google News RSS + yfinance.news, dedupe
│   │   ├── article.py        trafilatura full-text extraction, async, cached
│   │   └── store.py          SQLite: snapshots, articles, runs, findings
│   ├── analysis/
│   │   ├── screen.py         threshold + beta-residual mover detection
│   │   └── context.py        SPY/QQQ/sector ETF baseline, idiosyncratic move
│   ├── agents/
│   │   ├── graph.py          LangGraph supervisor: state, nodes, edges, fan-out
│   │   ├── triage.py         sub-agent: rank/select headlines
│   │   ├── analyst.py        sub-agent: attribute cause with citations
│   │   ├── verifier.py       sub-agent: drop unsupported claims
│   │   └── synthesizer.py    sub-agent: executive summary
│   ├── report/
│   │   ├── render.py         Jinja2 → HTML + Markdown
│   │   ├── sparkline.py      inline SVG from 30d closes (no JS libs)
│   │   └── templates/report.html.j2, report.md.j2
│   ├── publish.py            write run dir, update latest.html, open browser
│   └── cli.py                typer: run / watchlist / replay / doctor / watch
├── tests/                    unit + offline graph tests + fixtures/
├── data/                     SQLite db (gitignored)
└── reports/                  generated output (gitignored)
```

## Key component details

### Screening (`analysis/screen.py`) — deterministic, no LLM

A ticker is flagged when **any** condition holds (all configurable, per-ticker overridable):

- `|pct_change vs prev_close| >= move_threshold_pct` (default 3.0)
- `volume / avg_volume_30d >= volume_multiple` (default 2.0)
- `|open_gap_pct| >= gap_pct` (default 2.0)
- new 52-week high or low
- **`|beta-adjusted residual| >= idio_threshold`** (default 2.0) — the move that isn't explained
  by SPY. This is the highest-signal rule: it catches a 2% move on a flat market day that a
  raw threshold would miss, and suppresses a 3% move on a day the whole market fell 3%.

Beta is computed from 60 days of daily returns against SPY (`numpy.polyfit`), cached daily.
Only flagged tickers reach the LLM stage — this is the primary cost control.

### News gathering (`data/news.py`) — deterministic, async

Three sources, merged: Yahoo Finance RSS (`feeds.finance.yahoo.com/rss/2.0/headline?s=TICKER`),
Google News RSS (`news.google.com/rss/search?q=<company>+stock`), and `yfinance.Ticker.news`.
Both RSS feeds were verified live during planning.

- Filter to `published_at` within `lookback_hours` (default 36).
- Dedupe on canonicalized URL, then on fuzzy title match (`rapidfuzz.token_set_ratio > 88`).
- **Cross-run dedupe**: articles already cited in a previous run are marked `previously_reported`
  so the same story isn't re-reported every 30 minutes.
- Fetch full text for the top ~8 candidates with `trafilatura`, async via `httpx`, truncated to
  2500 chars, cached in SQLite by URL hash so reruns cost nothing.

### Sub-agent 1 — Triage (`agents/triage.py`), 1 call per mover

In: ticker, move stats as prose, up to 15 headlines with source + timestamp.
Out (schema-forced): `{"selected": [{"idx": int, "relevance": "high|medium|low", "why": str}],
"no_material_news": bool}`, max 4 selected. Short output → fast call.

### Sub-agent 2 — Analyst (`agents/analyst.py`), 1 call per mover

In: move stats, market context (SPY/sector move, residual), 2–4 article bodies.
Out (schema-forced):

```json
{"headline": "<=15 words", "narrative": "2-4 sentences",
 "driver": "company_news|earnings|analyst_action|sector|macro|no_identified_catalyst",
 "confidence": "high|medium|low",
 "evidence": [{"claim": "string", "source_idx": 0}],
 "watch_next": "one sentence"}
```

`source_idx` must index a supplied article. This citation requirement is what makes the next
step possible.

### Verifier (`agents/verifier.py`), 1 call total

- **Deterministic pre-pass** (no LLM): any `source_idx` out of range, or a claim on a ticker with
  `no_material_news`, is dropped immediately.
- **One batched LLM call** across all movers checks each surviving claim against its cited
  article snippet; unsupported claims are struck and that ticker's confidence is downgraded one
  level. This reflection pass is the main defense against a small model inventing causality, and
  costs one call regardless of watchlist size.

### Synthesizer (`agents/synthesizer.py`), 1 call

Produces only `{"summary": "3-5 sentences", "themes": [str], "top_story": ticker}`. All tables
and figures around it are rendered from data.

### LLM layer (`llm.py`)

`ChatOllama(base_url=..., model="qwen2.5:7b", temperature=0, num_ctx=16384)`.

**`num_ctx` must be set explicitly** — Ollama's default of 4096 would silently truncate article
text and produce confidently wrong output with no error.

`structured_call(schema, messages, retries=2)`: on `ValidationError`, retries with the validation
error appended to the prompt; after the final failure it returns a `Degraded` sentinel, records
the error in state, and the run continues with that ticker shown as "analysis unavailable". A
single bad JSON response must never abort the report. Per-call timeout (180s) and a global run
budget (15 min) after which remaining movers render as data-only cards. Every call is logged to
`llm_trace.jsonl` (prompt hash, tokens, duration) for tuning.

### Report (`report/render.py`)

Self-contained HTML, no external assets: run header with timestamp and market status, executive
summary, sortable movers table, per-ticker cards each with an inline SVG sparkline (generated in
Python from 30d closes), the causal narrative, and evidence lines as clickable source links with
timestamps and a confidence chip. A compact table lists unflagged watchlist names. A diagnostics
footer records model, duration, LLM call count, and any degraded tickers. Light/dark via
`prefers-color-scheme`. A Markdown twin is emitted for terminal reading and diffing.

Output: `reports/YYYY-MM-DD_HHMM/{report.html,report.md,state.json,llm_trace.jsonl}` plus a
`reports/latest.html` symlink; `--open` launches the browser.

`state.json` enables `fi-agent replay <run_id>`, which re-renders the report from stored data
with **zero LLM calls** — essential for iterating on the template without waiting minutes per run.

### Storage (`data/store.py`)

Stdlib `sqlite3`, no ORM. Tables: `snapshots` (per-ticker price history across runs, enabling
"vs last run" deltas), `articles` (text cache + cross-run dedupe), `runs`, `findings`.

### Scope guard

The report describes **what moved and what the news says about why**. Prompts explicitly forbid
buy/sell recommendations, price targets, and forward predictions; the template carries a
"monitoring tool, not investment advice" disclaimer. This keeps the deliverable a news-attribution
system rather than an advice engine.

## Build phases

Each phase is independently runnable and verifiable.

- **Phase 0 — Scaffold.** venv on 3.13, `requirements.txt`, config files, package skeleton,
  `fi-agent doctor`. Add `reports/`, `data/`, `config/local.yaml` to `.gitignore` (checked:
  `.venv` and `.env` are already ignored; these three are not).
- **Phase 1 — Data + screening, no LLM.** market/news/article/store, screen, context.
  `fi-agent run --no-llm` produces a data-only report end to end.
- **Phase 2 — LLM layer.** `llm.py`, `schemas.py`, triage and analyst sub-agents, exercised
  standalone against fixture articles.
- **Phase 3 — Graph.** LangGraph supervisor, `Send` fan-out, reducers, verifier, synthesizer,
  degradation paths.
- **Phase 4 — Report + publish.** Templates, sparklines, run dirs, `latest.html`, `replay`.
- **Phase 5 — Hardening.** Tests, `watch` mode with market-hours gating, README.

## Dependencies (`requirements.txt`)

Runtime: `langgraph`, `langchain-core`, `langchain-ollama`, `pydantic`, `pydantic-settings`,
`yfinance`, `feedparser`, `trafilatura`, `httpx`, `rapidfuzz`, `jinja2`, `pyyaml`, `typer`,
`rich`, `pandas`, `numpy`, `python-dateutil`.
Dev: `pytest`, `pytest-asyncio`, `respx`, `ruff`, `mypy`, `types-PyYAML`.

Per CLAUDE.md: venv at `.venv/` created with `/usr/local/bin/python3.13`, all installs inside it.

## Verification

```bash
/usr/local/bin/python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

1. **Environment** — `fi-agent doctor` confirms Ollama reachable, `qwen2.5:7b` present, Yahoo
   reachable, Python 3.13, venv active.
2. **Offline tests** — `pytest` runs the full graph against a `FakeLLM` returning canned valid
   JSON and recorded HTTP fixtures (`respx`). No network, no Ollama, sub-second. Unit tests cover
   screening math, beta residual, dedupe, schema validation, and SVG output.
3. **Data path, no LLM** — `fi-agent run --no-llm --open` renders a report from live market data;
   cross-check two tickers' quotes against Yahoo Finance in a browser.
4. **Single-ticker live run** — `fi-agent run --tickers NVDA --open` exercises both sub-agents and
   the verifier against the real model; inspect `llm_trace.jsonl` for call count and latency.
5. **Full live run** — `fi-agent run --open` on the real watchlist. Confirm: every numeric in the
   report matches `state.json`, every evidence claim has a working source link, runtime is within
   the ~3–5 min budget, and a deliberately unreachable Ollama URL degrades to a data-only report
   instead of crashing.
6. **Replay** — `fi-agent replay <run_id>` re-renders identically with zero LLM calls.

## Open items to confirm during Phase 0

- **The watchlist itself** — which tickers? I'll seed `config/watchlist.yaml` with a placeholder
  set and you edit it, unless you give me the list.
- **Run cadence** — I'm building `run` (on-demand) plus `watch --interval` (loop with market-hours
  gating). A launchd/cron entry can be added later if you want it fully unattended.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| 7B model reasoning quality | Narrow single-purpose prompts, JSON schema enforcement, enum confidence, citation requirement, verifier pass, numbers never model-generated |
| Server serializes inference → slow runs | Screening gates LLM work to flagged tickers only; ~10 calls/run budget; global time budget with graceful degradation |
| yfinance / Yahoo endpoint breakage | Three independent news sources; `doctor` detects it early; data layer isolated behind one module so a provider swap is contained |
| Article extraction failures (paywalls) | Fall back to RSS headline + summary; article count is best-effort, never fatal |
| Same story re-reported each run | Cross-run article dedupe in SQLite |
