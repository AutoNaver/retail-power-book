"""Settings from configs/*.toml and secrets from the environment."""

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.toml"
DEFAULT_ENV_FILE = REPO_ROOT / ".env"


class ConfigError(Exception):
    """A configuration file or secret is missing or invalid."""


@dataclass(frozen=True)
class DataConfig:
    cache_dir: Path


@dataclass(frozen=True)
class MarketConfig:
    bidding_zone: str


@dataclass(frozen=True)
class Config:
    data: DataConfig
    market: MarketConfig


def _require(raw: dict[str, Any], dotted_key: str, path: Path) -> Any:
    value: Any = raw
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ConfigError(f"missing key '{dotted_key}' in {path}")
        value = value[part]
    return value


def load_config(path: Path | None = None) -> Config:
    """Load a TOML config; relative paths resolve against the file's directory."""
    path = (path or DEFAULT_CONFIG_PATH).resolve()
    with path.open("rb") as f:
        raw = tomllib.load(f)
    cache_dir = Path(_require(raw, "data.cache_dir", path))
    if not cache_dir.is_absolute():
        cache_dir = (path.parent / cache_dir).resolve()
    return Config(
        data=DataConfig(cache_dir=cache_dir),
        market=MarketConfig(bidding_zone=str(_require(raw, "market.bidding_zone", path))),
    )


def entsoe_token(env_file: Path | None = None) -> str:
    """ENTSO-E API token from the environment, falling back to the `.env` file.

    The process environment takes precedence. The token is never logged.
    """
    name = "ENTSOE_API_TOKEN"
    token = os.environ.get(name) or dotenv_values(env_file or DEFAULT_ENV_FILE).get(name)
    if not token:
        raise ConfigError(f"{name} is not set; add it to the environment or to .env")
    return token
