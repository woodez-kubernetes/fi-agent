# Deploy & Run

How to set up fi-agent and keep it running unattended on this Mac.

Everything below assumes the project lives at:

```
/Users/kwood/projects/coding-repos/fi-agent
```

---

## Before you start

Two things must be true:

1. **The Ollama machine is on and reachable** at `http://192.168.2.167:11434` with the
   `qwen2.5:7b` model pulled.
2. **This Mac stays awake.** A sleeping Mac runs nothing. Open
   System Settings → Lock Screen and set the display to turn off, but in
   System Settings → Battery (or Energy Saver) enable
   **"Prevent automatic sleeping when the display is off"**.

Check the Ollama box is up:

```bash
curl -s http://192.168.2.167:11434/api/tags
```

If that prints JSON listing `qwen2.5:7b`, you are good. If it hangs or errors, start
Ollama on that machine first — nothing else here will work.

---

## Part 1 — One-time setup

Run these once. Takes about two minutes.

```bash
cd /Users/kwood/projects/coding-repos/fi-agent
```

Create the virtual environment on Python 3.13:

```bash
/usr/local/bin/python3.13 -m venv .venv
```

Install everything into it:

```bash
.venv/bin/python -m pip install -r requirements.txt && .venv/bin/python -m pip install -e .
```

Confirm every dependency is reachable:

```bash
.venv/bin/fi-agent doctor
```

You want all seven rows to say `ok`:

```
Check                  Detail
Python 3.13        ok  3.13.0
Virtualenv active  ok  /Users/kwood/projects/coding-repos/fi-agent/.venv
Ollama             ok  qwen2.5:7b available at http://192.168.2.167:11434
Market data        ok  SPY 769.35 (-0.23%)
News feeds         ok  15 headlines for NVDA
Watchlist          ok  10 tickers
Market session     ok  closed
```

If any row says `fail`, fix that before continuing — see
[Troubleshooting](#troubleshooting).

---

## Part 2 — Prove it works

Do one full run by hand and open the report:

```bash
.venv/bin/fi-agent run --open
```

This takes about 75 seconds. Your browser opens the finished report.

If you want a fast check without waiting on the model, this one takes ~10 seconds and
uses no LLM at all:

```bash
.venv/bin/fi-agent run --no-llm --open
```

---

## Part 3 — Email alerts (optional)

Get the report emailed to you when something new starts moving.

**The rule:** it emails when a ticker flags that was *not* flagged in the previous run.
If NVDA stays flagged all afternoon you get one email, not one every 30 minutes.

### Step 1 — Create a Gmail App Password

Your normal Google password will not work; Gmail requires an App Password for SMTP.

1. Turn on 2-Step Verification at <https://myaccount.google.com/security> if it isn't on.
2. Go to <https://myaccount.google.com/apppasswords>.
3. Create one named `fi-agent`.
4. Google shows a 16-character password. Copy it.

### Step 2 — Put it in a .env file

```bash
cp /Users/kwood/projects/coding-repos/fi-agent/.env.example /Users/kwood/projects/coding-repos/fi-agent/.env
```

Open `.env` and paste the 16 characters after the `=`, with no spaces or quotes:

```
FI_AGENT_SMTP_PASSWORD=abcdefghijklmnop
```

`.env` is gitignored, so it will never be committed. Nothing else in the project stores
this value and it is never written to the logs.

### Step 3 — Turn email on

In `config/settings.yaml`, change `enabled` to `true`:

```yaml
email:
  enabled: true
  to: ["kevin.wood75@gmail.com"]
  from_address: "kevin.wood75@gmail.com"
```

### Step 4 — Send yourself a test

First check the connection with a one-line message:

```bash
.venv/bin/fi-agent email-test
```

If it arrives, SMTP works. If it fails, the error says what to fix — the usual cause is
using the account password instead of an App Password.

Then email yourself a real report:

```bash
.venv/bin/fi-agent run --force-email
```

`--force-email` sends whatever this run flags, ignoring the newly-flagged rule. Without
it a test run usually sends nothing, because the same names were already flagged last
time — which is the rule working correctly, but makes for a confusing test.

Confirm the configuration is seen:

```bash
.venv/bin/fi-agent doctor
```

The `Email` row should read `to kevin.wood75@gmail.com via smtp.gmail.com`.

### Controlling it per run

Force an email for one run, ignoring the config:

```bash
.venv/bin/fi-agent run --email
```

Suppress it for one run:

```bash
.venv/bin/fi-agent run --no-email
```

Email the current picture on demand, whether or not anything is newly flagged:

```bash
.venv/bin/fi-agent run --force-email
```

### A note on how the email looks

The body is the full report. **Gmail strips SVG**, so the sparklines will not appear in
the email — everything else does. To also receive the file itself, which opens with the
sparklines intact, set `attach_report: true` in `config/settings.yaml`.

---

## Part 4 — Keep it running

This installs a **LaunchAgent**: macOS starts the agent when you log in, restarts it if
it ever dies, and keeps it running in the background. You do not need a terminal open.

It runs every 30 minutes and **skips cycles automatically when the market is closed**, so
it is safe to leave on permanently.

Create the log folder:

```bash
mkdir -p /Users/kwood/projects/coding-repos/fi-agent/logs
```

Install the job:

```bash
cp /Users/kwood/projects/coding-repos/fi-agent/deploy/com.fi-agent.watch.plist ~/Library/LaunchAgents/
```

Start it:

```bash
launchctl load -w ~/Library/LaunchAgents/com.fi-agent.watch.plist
```

That's it. It is now running and will come back on every reboot.

Confirm it started:

```bash
launchctl list | grep fi-agent
```

You should see a line like `-  0  com.fi-agent.watch`. The middle number is the last exit
code — `0` is healthy.

---

## Part 5 — Day-to-day control

### Watch it work

```bash
tail -f /Users/kwood/projects/coding-repos/fi-agent/logs/fi-agent.log
```

Press `Ctrl-C` to stop watching (this does not stop the agent).

### Stop it

```bash
launchctl unload ~/Library/LaunchAgents/com.fi-agent.watch.plist
```

### Start it again

```bash
launchctl load -w ~/Library/LaunchAgents/com.fi-agent.watch.plist
```

### Restart it (after changing settings)

```bash
launchctl unload ~/Library/LaunchAgents/com.fi-agent.watch.plist && launchctl load -w ~/Library/LaunchAgents/com.fi-agent.watch.plist
```

### Remove it permanently

```bash
launchctl unload ~/Library/LaunchAgents/com.fi-agent.watch.plist && rm ~/Library/LaunchAgents/com.fi-agent.watch.plist
```

---

## Part 6 — Where the reports go

Every run writes a timestamped folder under `reports/`:

```
reports/
├── latest.html                  ← always the most recent report
└── 2026-08-31_030017/
    ├── report.html              ← the report, open this
    ├── report.md                ← same thing as text
    ├── state.json               ← raw data, used by `replay`
    └── llm_trace.jsonl          ← timing of each model call
```

Open the newest report any time:

```bash
open /Users/kwood/projects/coding-repos/fi-agent/reports/latest.html
```

**Tip:** drag `reports/latest.html` into your browser's bookmarks bar. The link always
points at the newest report, so one click gives you the current picture.

---

## Part 7 — Changing what it watches

### Add or remove a ticker

```bash
.venv/bin/fi-agent watchlist add COIN --name Coinbase --sector-etf XLF
```

```bash
.venv/bin/fi-agent watchlist remove XOM
```

```bash
.venv/bin/fi-agent watchlist list
```

### Change how often it runs, or anything else

Edit `deploy/com.fi-agent.watch.plist`, change the `30` under `--interval` to the number
of minutes you want, then copy and restart:

```bash
cp /Users/kwood/projects/coding-repos/fi-agent/deploy/com.fi-agent.watch.plist ~/Library/LaunchAgents/ && launchctl unload ~/Library/LaunchAgents/com.fi-agent.watch.plist && launchctl load -w ~/Library/LaunchAgents/com.fi-agent.watch.plist
```

### Change thresholds or the Ollama address

Edit `config/settings.yaml`, then restart the agent (see Part 5). The most likely edits:

| Setting | Meaning |
|---|---|
| `llm.base_url` | Address of the Ollama machine |
| `screening.move_threshold_pct` | How big a move must be to get investigated (default 3%) |
| `screening.idio_threshold_pct` | How much unexplained move triggers a look (default 2%) |
| `screening.volume_multiple` | How much heavier than normal volume must be to flag (default 2x) |
| `news.lookback_hours` | How far back to search for news (default 36) |

Changes take effect on the next cycle after a restart.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `doctor` says Ollama `fail` | The LLM machine is off, asleep, or moved to a new IP | Start Ollama; if the IP changed, update `llm.base_url` in `config/settings.yaml` |
| `launchctl list` shows a non-zero exit code | It crashed on startup | Read `logs/fi-agent.err.log` |
| No new reports appearing | The Mac slept, or the market is closed | Check `logs/fi-agent.log`; "Market closed, skipping this cycle" is normal and expected outside 09:30–16:00 ET on weekdays |
| `429 Too Many Requests` in the log | Yahoo is rate-limiting the RSS feed | Harmless. Google News and yfinance cover it; the report still gets its news |
| Reports say "no identified catalyst" a lot | Working as designed | The agent refuses to assert a cause it cannot cite. Outside market hours there is often genuinely no fresh news |
| No emails arriving | Nothing *newly* flagged | By design it only mails when a name flags that wasn't flagged last run. The run output says `no newly flagged tickers since the last run`. Use `fi-agent run --force-email` to send anyway |
| Email says `FI_AGENT_SMTP_PASSWORD is not set` | No `.env`, or the variable is blank | See Part 3, Step 2 |
| Email says SMTP rejected the login | Using the Google account password | It must be a 16-character App Password. See Part 3, Step 1 |
| Sparklines missing from the email | Gmail strips SVG | Expected. Set `attach_report: true` and open the attachment |
| Everything looks broken | — | Run `.venv/bin/fi-agent doctor` first; it isolates which layer failed |

### Nothing is working and I want to start over

```bash
launchctl unload ~/Library/LaunchAgents/com.fi-agent.watch.plist
```

```bash
rm -rf /Users/kwood/projects/coding-repos/fi-agent/.venv
```

Then repeat Part 1.

---

## Alternative: run it in a terminal instead

If you would rather not install a background job, just run it in a terminal window and
leave the window open:

```bash
cd /Users/kwood/projects/coding-repos/fi-agent && .venv/bin/fi-agent watch --interval 30
```

Press `Ctrl-C` to stop. It stops when you close the window or the Mac sleeps, which is
why the LaunchAgent in Part 4 is the better option for leaving it running.
