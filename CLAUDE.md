# fi-agent

Playing with AI agents.

## Rules

### Language & stack

- All code is written in Python. Target **Python 3.13**.
- Do not introduce another language for application code without asking first.

### Virtual environment

- The virtual environment lives at `.venv/` in the project root.
- All code runs inside that virtual environment. Never run project code against
  the system Python.
- Activate before running anything — scripts, tests, tooling:
  ```bash
  source .venv/bin/activate
  ```
- `pip install` is only ever run with the venv active. Never install packages
  globally or with `--user`.
- `.venv/` is git-ignored and must never be committed.
- Dependencies are tracked in `requirements.txt`. Add new packages there after
  installing them.

### Workflow

- Always present a plan and get my approval before executing it. This applies to
  writing code, editing files, running commands that change state, and
  installing packages.
- Read-only exploration (reading files, searching, `git status`) does not need
  approval.
