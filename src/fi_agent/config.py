"""Configuration loading: YAML files plus optional local overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


def project_root() -> Path:
    """Repo root, i.e. the directory containing config/ and reports/."""
    return Path(__file__).resolve().parents[2]


class LLMSettings(BaseModel):
    base_url: str = "http://192.168.2.167:11434"
    model: str = "qwen2.5:7b"
    num_ctx: int = 16384
    temperature: float = 0.0
    request_timeout_s: int = 180
    max_retries: int = 2


class ScreeningSettings(BaseModel):
    move_threshold_pct: float = 3.0
    volume_multiple: float = 2.0
    gap_pct: float = 2.0
    idio_threshold_pct: float = 2.0
    flag_52w_extremes: bool = True
    benchmark: str = "SPY"
    beta_lookback_days: int = 60


class NewsSettings(BaseModel):
    lookback_hours: int = 36
    max_headlines_per_ticker: int = 15
    max_articles_per_ticker: int = 4
    fetch_concurrency: int = 8
    article_char_limit: int = 2500
    fuzzy_title_threshold: int = 88
    http_timeout_s: int = 20
    user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )


class RunSettings(BaseModel):
    budget_seconds: int = 900
    max_movers: int = 8


class PathSettings(BaseModel):
    reports_dir: str = "reports"
    database: str = "data/fi_agent.db"


class Settings(BaseModel):
    llm: LLMSettings = Field(default_factory=LLMSettings)
    screening: ScreeningSettings = Field(default_factory=ScreeningSettings)
    news: NewsSettings = Field(default_factory=NewsSettings)
    run: RunSettings = Field(default_factory=RunSettings)
    paths: PathSettings = Field(default_factory=PathSettings)

    @property
    def reports_path(self) -> Path:
        return _resolve(self.paths.reports_dir)

    @property
    def database_path(self) -> Path:
        return _resolve(self.paths.database)


class TickerConfig(BaseModel):
    symbol: str
    name: str = ""
    sector_etf: str | None = None
    move_threshold_pct: float | None = None
    aliases: list[str] = Field(
        default_factory=list,
        description="Extra names the press uses, e.g. Google for Alphabet",
    )

    def display_name(self) -> str:
        return self.name or self.symbol


class Watchlist(BaseModel):
    tickers: list[TickerConfig] = Field(default_factory=list)

    def symbols(self) -> list[str]:
        return [t.symbol for t in self.tickers]

    def get(self, symbol: str) -> TickerConfig | None:
        return next((t for t in self.tickers if t.symbol == symbol), None)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root() / path


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_settings(config_dir: Path | None = None) -> Settings:
    """Load config/settings.yaml, overlaid with config/local.yaml if present."""
    directory = config_dir or project_root() / "config"
    data = _deep_merge(
        _read_yaml(directory / "settings.yaml"), _read_yaml(directory / "local.yaml")
    )
    return Settings.model_validate(data)


def watchlist_path(config_dir: Path | None = None) -> Path:
    directory = config_dir or project_root() / "config"
    return directory / "watchlist.yaml"


def load_watchlist(config_dir: Path | None = None) -> Watchlist:
    return Watchlist.model_validate(_read_yaml(watchlist_path(config_dir)))


def save_watchlist(watchlist: Watchlist, config_dir: Path | None = None) -> Path:
    path = watchlist_path(config_dir)
    payload = {
        "tickers": [
            t.model_dump(exclude_none=True, exclude_defaults=False) for t in watchlist.tickers
        ]
    }
    with path.open("w") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)
    return path
