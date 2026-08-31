# fi-agent

A multi-agent stock watchlist monitor that runs entirely on a local Qwen LLM. It watches
a list of tickers, detects moves that are actually worth explaining, hunts for news that
accounts for them, checks its own citations, and publishes an HTML + Markdown report.

No cloud LLM, no API keys, no paid data feeds.

## How it works

```
load config → fetch market data → screen → market context
                                              │
                                  fan out over flagged movers
                          ┌───────────────────┼───────────────────┐
                     gather news         gather news         gather news    (no LLM)
                     triage              triage              triage         (1 call each)
                     analyse             analyse             analyse        (1 call each)
                          └───────────────────┼───────────────────┘
                                          verify        (1 batched call)
                                          summarise     (1 call)
                                          render        (no LLM)
                                          publish
```

Three principles hold the design together, all of them consequences of running a 4-bit
7B model:

**The LLM never touches a number.** Every price, percentage, volume ratio and beta
residual is computed in Python and rendered straight into the report. The model only
ranks headlines, attributes causes from prose, and writes summary text. Numeric
hallucination is structurally impossible rather than merely unlikely.

**Every model response is schema-validated, using enums rather than floats.** Asked for a
0.0–1.0 confidence, qwen2.5:7b returns `100`. Asked to choose between `high`, `medium`
and `low`, it complies. Generation is constrained by the Pydantic model's JSON schema via
Ollama's `format` parameter, validated on receipt, and retried with the validation error
fed back.

**Claims must cite their source, and the citations are checked.** The analyst must attach
each claim to the index of an article it was given. A deterministic pass drops citations
that point nowhere, then one batched LLM call checks each surviving claim against the
article it cites and strikes the ones the article does not support. When nothing
survives, the report says so instead of inventing a cause.

### Screening decides what the model ever sees

A ticker is investigated only if it clears at least one threshold: a raw percentage move,
a volume spike, an opening gap, a 52-week extreme, or — the highest-signal rule — a
**beta-adjusted residual** against the benchmark.

The residual is what makes the report worth reading. A raw threshold answers "did this
move a lot?", which on a day the whole market fell 3% flags everything and explains
nothing. The residual answers "did this move for reasons of its own?" In a live run it
flagged GOOGL on a +1.74% day, which no 3% threshold would have caught, because +2.04%
of that move was unexplained by SPY.

## Setup

```bash
python3.13 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && pip install -e .
```

Point it at your Ollama server in `config/settings.yaml`:

```yaml
llm:
  base_url: "http://192.168.2.167:11434"
  model: "qwen2.5:7b"
  num_ctx: 16384
```

`num_ctx` matters. Ollama defaults to 4096, which silently truncates article text and
produces confident nonsense with no error anywhere.

Then check everything is reachable:

```bash
fi-agent doctor
```

## Usage

```bash
fi-agent run --open
```

Other commands:

| Command | What it does |
|---|---|
| `fi-agent run` | Full run: fetch, screen, investigate, publish |
| `fi-agent run --no-llm` | Data and screening only, no agents. Fast. |
| `fi-agent run -t NVDA,AMD` | Restrict to specific tickers, on or off the watchlist |
| `fi-agent replay latest` | Re-render a stored run with current templates, **zero LLM calls** |
| `fi-agent watch -i 30` | Run every 30 minutes during market hours |
| `fi-agent doctor` | Check Ollama, market data, news feeds, venv |
| `fi-agent watchlist list/add/remove` | Manage `config/watchlist.yaml` |

`replay` is the one to know about: it rebuilds the report from the saved `state.json`, so
templates can be reworked without spending minutes of inference regenerating content that
has not changed.

## Output

Each run writes `reports/YYYY-MM-DD_HHMMSS/`:

| File | Contents |
|---|---|
| `report.html` | Self-contained report — inline SVG sparklines, light/dark, no external assets |
| `report.md` | Same content for terminal reading and diffing |
| `state.json` | Everything needed to replay the run |
| `llm_trace.jsonl` | Per-call latency and token counts |

`reports/latest.html` points at the most recent run.

## Configuration

`config/watchlist.yaml`:

```yaml
tickers:
  - symbol: NVDA
    name: NVIDIA
    sector_etf: SMH          # separates sector moves from company moves
    move_threshold_pct: 4.5  # raise it for high-beta names
  - symbol: GOOGL
    name: Alphabet
    sector_etf: XLC
    aliases: [Google]        # names the press actually uses
```

`config/settings.yaml` holds LLM connection details, screening thresholds, news lookback
and paths. Copy it to `config/local.yaml` to override anything without touching git.

## Data sources

Prices and history come from Yahoo Finance via `yfinance`, in one batched daily-bar
download per run. News is merged from Yahoo Finance RSS, Google News RSS and yfinance's
own feed, then deduplicated by canonical URL and fuzzy headline match.

A deterministic relevance filter runs before any model sees a headline. Yahoo's
per-ticker feeds mix in unrelated market stories — a request for NVDA news returns pieces
on CVS and Bitcoin — and screening that out in Python is both cheaper and more reliable
than asking a 7B model why a CVS story is in front of it.

## Performance

Roughly `2 × movers + 2` LLM calls per run. A typical run on a 10-name watchlist with 3
movers is 8 calls and finishes in about 75 seconds.

If your Ollama server serialises inference, wall-clock time is the sum of the calls;
screening keeping the mover list short is what keeps runs bounded. The fan-out still pays
for itself because RSS fetching and article extraction genuinely overlap.

## Failure behaviour

Nothing in a run is allowed to abort the report:

| Failure | Result |
|---|---|
| Ollama unreachable | Data-only report, prices and screening intact |
| Model returns invalid JSON | Retried with the error, then that ticker degrades to data-only |
| News feed 429s or dies | Other sources carry it; a total failure degrades one ticker |
| Article paywalled | Falls back to the RSS summary, marked "headline only" in the report |
| Run exceeds its time budget | Remaining movers render as data-only cards |
| Verification pass fails | Claims retained, noted in diagnostics |

## Development

```bash
pytest              # 87 tests, fully offline, under a second
ruff check src tests
mypy
```

The suite stubs the model with a `FakeLLM` and the network with fixtures, so the whole
graph — fan-out, reducers, every degradation path — is exercised without touching Ollama
or the internet.

## Scope

This tool describes what moved and what the reporting says caused it. Prompts forbid
buy/sell recommendations, price targets and forward predictions. Narrative text is
model-written and can be wrong; the citations are there so you can check it. It is a news
monitoring tool, not investment advice.
