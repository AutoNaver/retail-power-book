from pathlib import Path

import pytest

from rpb.config import ConfigError, entsoe_token, load_config

TOKEN_VAR = "ENTSOE_API_TOKEN"


def write_config(directory: Path, text: str) -> Path:
    path = directory / "test.toml"
    path.write_text(text)
    return path


def test_values_and_relative_path_resolution(tmp_path: Path) -> None:
    path = write_config(
        tmp_path, '[data]\ncache_dir = "../cache"\n[market]\nbidding_zone = "DE-LU"\n'
    )
    config = load_config(path)
    assert config.data.cache_dir == (tmp_path.parent / "cache").resolve()
    assert config.market.bidding_zone == "DE-LU"


def test_absolute_path_is_kept(tmp_path: Path) -> None:
    cache = tmp_path / "abs"
    path = write_config(
        tmp_path, f'[data]\ncache_dir = "{cache.as_posix()}"\n[market]\nbidding_zone = "DE-LU"\n'
    )
    assert load_config(path).data.cache_dir == cache


def test_missing_key_names_the_key(tmp_path: Path) -> None:
    path = write_config(tmp_path, '[data]\ncache_dir = "cache"\n')
    with pytest.raises(ConfigError, match="market.bidding_zone"):
        load_config(path)


def test_default_config_loads() -> None:
    config = load_config()
    assert config.market.bidding_zone == "DE-LU"
    assert config.data.cache_dir.is_absolute()


def test_missing_token_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TOKEN_VAR, raising=False)
    with pytest.raises(ConfigError, match=TOKEN_VAR):
        entsoe_token(tmp_path / "missing.env")


def test_empty_token_in_env_file_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TOKEN_VAR, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(f"{TOKEN_VAR}=\n")
    with pytest.raises(ConfigError):
        entsoe_token(env_file)


def test_token_from_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TOKEN_VAR, "from-env")
    env_file = tmp_path / ".env"
    env_file.write_text(f"{TOKEN_VAR}=from-file\n")
    assert entsoe_token(env_file) == "from-env"


def test_token_from_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TOKEN_VAR, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(f"{TOKEN_VAR}=from-file\n")
    assert entsoe_token(env_file) == "from-file"
