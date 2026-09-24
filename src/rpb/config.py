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
SUPPORTED_BIDDING_ZONES = ("DE-LU",)  # scope: see AGENTS.md, Domain rules


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


def _require_str(raw: dict[str, Any], dotted_key: str, path: Path) -> str:
    value = _require(raw, dotted_key, path)
    if not isinstance(value, str):
        raise ConfigError(f"'{dotted_key}' in {path} must be a string, got {value!r}")
    return value


def load_config(path: Path | None = None) -> Config:
    """Load a TOML config; relative paths resolve against the file's directory.

    The default is `configs/default.toml` in the source checkout. The project is run
    from a checkout (`uv sync`), not installed as a wheel; outside a checkout, pass
    `path` explicitly.
    """
    path = (path or DEFAULT_CONFIG_PATH).resolve()
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}; pass the path to a config explicitly")
    with path.open("rb") as f:
        raw = tomllib.load(f)
    cache_dir = Path(_require_str(raw, "data.cache_dir", path))
    if not cache_dir.is_absolute():
        cache_dir = (path.parent / cache_dir).resolve()
    bidding_zone = _require_str(raw, "market.bidding_zone", path)
    if bidding_zone not in SUPPORTED_BIDDING_ZONES:
        raise ConfigError(
            f"unsupported bidding zone {bidding_zone!r} in {path}; "
            f"supported: {', '.join(SUPPORTED_BIDDING_ZONES)}"
        )
    return Config(
        data=DataConfig(cache_dir=cache_dir),
        market=MarketConfig(bidding_zone=bidding_zone),
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
