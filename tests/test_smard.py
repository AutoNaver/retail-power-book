import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rpb.data.smard import (
    load_day_ahead_prices,
    parse_chart_data,
    quarter_hours_to_hourly,
)

FIXTURES = Path(__file__).parent / "fixtures" / "smard"
# Week of Monday 2025-10-20 Europe/Berlin: DST ends on Sunday 2025-10-26, and it has negative prices.
WEEK_MS = 1760911200000
QUARTER_HOUR_FILE = FIXTURES / f"4169_DE-LU_quarterhour_{WEEK_MS}.json"
HOUR_FILE = FIXTURES / f"4169_DE-LU_hour_{WEEK_MS}.json"


def berlin(ts: str) -> pd.Timestamp:
    return pd.Timestamp(ts, tz="Europe/Berlin")


class FakeSmard:
    """Serves the committed fixture in place of the SMARD API and counts requests."""

    def __init__(self, chunk: bytes = QUARTER_HOUR_FILE.read_bytes()) -> None:
        self.chunk = chunk
        self.urls: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.urls.append(url)
        if url.endswith("index_quarterhour.json"):
            return json.dumps({"timestamps": [WEEK_MS]}).encode()
        if url.endswith(f"_quarterhour_{WEEK_MS}.json"):
            return self.chunk
        raise AssertionError(f"unexpected URL {url}")


def test_parse_fixture_is_utc_quarter_hours() -> None:
    series = parse_chart_data(json.loads(QUARTER_HOUR_FILE.read_text()))
    assert str(series.index.tz) == "UTC"
    assert series.index[0] == berlin("2025-10-20 00:00")
    assert len(series) == 169 * 4  # 7 days, one of them 25 hours
    assert (series.index.to_series().diff().dropna() == pd.Timedelta(minutes=15)).all()


def test_fixture_aggregates_to_smard_hourly_values_and_keeps_negatives() -> None:
    hourly = quarter_hours_to_hourly(parse_chart_data(json.loads(QUARTER_HOUR_FILE.read_text())))
    smard_hourly = parse_chart_data(json.loads(HOUR_FILE.read_text()))

    assert hourly.index.equals(smard_hourly.index)
    # SMARD publishes hourly means rounded to 2 decimals.
    np.testing.assert_allclose(hourly, smard_hourly, atol=0.005 + 1e-9)
    # 12 negative hours. SMARD's hourly file shows 11: it rounds the mean at
    # 2025-10-25 01:00 UTC (quarter-hours 0, 0, 0, -0.01 -> -0.0025) to 0.00.
    assert (hourly < 0).sum() == 12
    assert hourly.min() < -0.5
    local_days = hourly.index.tz_convert("Europe/Berlin").date
    assert (local_days == pd.Timestamp("2025-10-26").date()).sum() == 25


@pytest.mark.parametrize(("day", "hours"), [("2024-03-31", 23), ("2024-10-27", 25)])
def test_dst_day_quarter_hours_aggregate_to_known_means(day: str, hours: int) -> None:
    start, end = berlin(day), berlin(day) + pd.DateOffset(days=1)
    index = pd.date_range(start, end, freq="15min", inclusive="left").tz_convert("UTC")
    assert len(index) == hours * 4
    # Quarter-hour values h, h+1, h+2, h+3 in hour h: the hourly mean is h + 1.5.
    hour_number = np.repeat(np.arange(hours), 4)
    values = hour_number + np.tile(np.arange(4), hours) - 40.0  # shifted to include negatives
    hourly = quarter_hours_to_hourly(pd.Series(values, index=index, name="price_eur_mwh"))

    assert len(hourly) == hours
    np.testing.assert_allclose(hourly.to_numpy(), np.arange(hours) + 1.5 - 40.0)
    assert hourly.name == "price_eur_mwh"


def test_incomplete_hours_are_nan_not_partial_means() -> None:
    index = pd.date_range("2025-11-03 00:00", periods=12, freq="15min", tz="UTC")
    values = np.arange(12.0)
    values[5] = np.nan  # hour 1 has a NaN quarter-hour
    series = pd.Series(values, index=index).drop(index[9])  # hour 2 is missing a row
    hourly = quarter_hours_to_hourly(series)

    assert hourly.iloc[0] == pytest.approx(1.5)
    assert np.isnan(hourly.iloc[1])
    assert np.isnan(hourly.iloc[2])


def test_aggregation_rejects_naive_index() -> None:
    naive = pd.Series(1.0, index=pd.date_range("2025-11-03", periods=4, freq="15min"))
    with pytest.raises(ValueError, match="UTC"):
        quarter_hours_to_hourly(naive)


def test_load_selects_range_and_fills_gaps_with_nan(tmp_path: Path) -> None:
    fetch = FakeSmard()
    start, end = berlin("2025-10-26 00:00"), berlin("2025-10-28 00:00")
    prices = load_day_ahead_prices(start, end, tmp_path, fetch=fetch)

    assert str(prices.index.tz) == "UTC"
    assert len(prices) == 25 + 24
    assert prices.name == "price_eur_mwh"
    # The fixture week ends at Monday 2025-10-27 00:00 local; SMARD has no data after it here.
    on_sunday = prices.index < berlin("2025-10-27 00:00")
    assert prices[on_sunday].notna().all()
    assert prices[~on_sunday].isna().all()


def test_weekly_files_are_cached(tmp_path: Path) -> None:
    start, end = berlin("2025-10-21 00:00"), berlin("2025-10-22 00:00")
    first = FakeSmard()
    load_day_ahead_prices(start, end, tmp_path, fetch=first)
    assert (tmp_path / "smard" / QUARTER_HOUR_FILE.name).exists()

    second = FakeSmard(chunk=b"not used")
    prices = load_day_ahead_prices(start, end, tmp_path, fetch=second)
    assert second.urls == [first.urls[0]]  # only the index is fetched again
    assert prices.notna().all()


def test_week_still_being_published_is_not_cached(tmp_path: Path) -> None:
    payload = json.loads(QUARTER_HOUR_FILE.read_text())
    payload["series"][-8:] = [[ms, None] for ms, _ in payload["series"][-8:]]
    fetch = FakeSmard(chunk=json.dumps(payload).encode())

    prices = load_day_ahead_prices(
        berlin("2025-10-26 00:00"), berlin("2025-10-27 00:00"), tmp_path, fetch=fetch
    )
    assert not (tmp_path / "smard").exists()
    assert prices.iloc[-2:].isna().all()
    assert prices.iloc[:-2].notna().all()


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (pd.Timestamp("2025-10-21 00:00"), pd.Timestamp("2025-10-22 00:00")),
        (berlin("2025-10-21 00:30"), berlin("2025-10-22 00:00")),
        (berlin("2025-10-22 00:00"), berlin("2025-10-21 00:00")),
    ],
)
def test_invalid_ranges_raise(start: pd.Timestamp, end: pd.Timestamp, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_day_ahead_prices(start, end, tmp_path, fetch=FakeSmard())


@pytest.mark.parametrize("utc_hour", ["2025-10-26 00:00", "2025-10-26 01:00"])
def test_range_can_start_on_either_ambiguous_autumn_hour(utc_hour: str, tmp_path: Path) -> None:
    # Both are 02:00 Europe/Berlin on 2025-10-26, first at +02:00, then at +01:00.
    start = pd.Timestamp(utc_hour, tz="UTC").tz_convert("Europe/Berlin")
    end = berlin("2025-10-26 04:00")
    prices = load_day_ahead_prices(start, end, tmp_path, fetch=FakeSmard())

    assert prices.index[0] == start
    assert len(prices) == (end - start) // pd.Timedelta(hours=1)
    assert prices.notna().all()


def test_aggregation_accepts_utc_alias_and_returns_utc() -> None:
    index = pd.date_range("2025-11-03 00:00", periods=8, freq="15min", tz="Etc/UTC")
    hourly = quarter_hours_to_hourly(pd.Series(np.arange(8.0), index=index))
    assert str(hourly.index.tz) == "UTC"
    np.testing.assert_allclose(hourly.to_numpy(), [1.5, 5.5])


def test_aggregation_rejects_nat_timestamps() -> None:
    index = pd.DatetimeIndex(["2025-11-03 00:00", pd.NaT], tz="UTC")
    with pytest.raises(ValueError, match="NaT"):
        quarter_hours_to_hourly(pd.Series([1.0, 2.0], index=index))
