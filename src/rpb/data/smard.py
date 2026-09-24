"""DE-LU day-ahead prices from SMARD.

Data source: Bundesnetzagentur | SMARD.de, licensed under CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/).

SMARD serves chart data in weekly files, each starting Monday 00:00 Europe/Berlin:

    {BASE_URL}/{filter}/{region}/index_{resolution}.json          -> {"timestamps": [...]}
    {BASE_URL}/{filter}/{region}/{filter}_{region}_{resolution}_{timestamp}.json
                                                                  -> {"series": [[ms, value], ...]}

Timestamps are epoch milliseconds (UTC) of interval starts; missing values are null.
Since delivery day 2025-10-01 the day-ahead auction clears in 15-minute products.
This loader always fetches quarter-hours (before that date SMARD repeats each
hourly price four times) and aggregates them to hours in one place.
"""

import json
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rpb.timeutils import require_utc

BASE_URL = "https://www.smard.de/app/chart_data"
DAY_AHEAD_FILTER = 4169
REGION = "DE-LU"
RESOLUTION = "quarterhour"
QUARTER_HOURS_PER_HOUR = 4

Fetch = Callable[[str], bytes]


def _http_get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "retail-power-book"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def parse_chart_data(payload: dict[str, Any], name: str = "price_eur_mwh") -> pd.Series:
    """Series of a SMARD chart-data payload at its native resolution; nulls become NaN."""
    series = payload["series"]
    ms = np.array([point[0] for point in series], dtype="int64")
    values = np.array([np.nan if point[1] is None else point[1] for point in series], dtype=float)
    index = pd.DatetimeIndex(pd.to_datetime(ms, unit="ms", utc=True), name="delivery_start_utc")
    return pd.Series(values, index=index, name=name)


def quarter_hours_to_hourly(quarter_hours: pd.Series) -> pd.Series:
    """Hourly mean of four quarter-hour prices: the price of a flat 1 MW hour.

    The result covers every hour from the first to the last input hour. An hour with
    fewer than four valid quarter-hours, including one with no rows at all, is NaN,
    not a partial mean or a gap in the index.
    """
    if not isinstance(quarter_hours.index, pd.DatetimeIndex):
        raise TypeError("quarter-hour series must have a DatetimeIndex")
    require_utc(quarter_hours.index)
    index = quarter_hours.index.tz_convert("UTC")
    if (index.minute % 15 != 0).any() or (index.second != 0).any():
        raise ValueError("quarter-hour timestamps must fall on 15-minute boundaries")
    grouped = quarter_hours.set_axis(index).groupby(index.floor("h"))
    hourly = grouped.mean().where(grouped.count() == QUARTER_HOURS_PER_HOUR)
    if len(hourly):
        hourly = hourly.reindex(pd.date_range(hourly.index[0], hourly.index[-1], freq="h"))
    hourly.index.name = "delivery_start_utc"
    return hourly.rename(quarter_hours.name)


def _url(timestamp_ms: int | None = None) -> str:
    prefix = f"{BASE_URL}/{DAY_AHEAD_FILTER}/{REGION}"
    if timestamp_ms is None:
        return f"{prefix}/index_{RESOLUTION}.json"
    return f"{prefix}/{DAY_AHEAD_FILTER}_{REGION}_{RESOLUTION}_{timestamp_ms}.json"


def _load_chunk(timestamp_ms: int, cache_dir: Path, fetch: Fetch) -> dict[str, Any]:
    path = cache_dir / "smard" / f"{DAY_AHEAD_FILTER}_{REGION}_{RESOLUTION}_{timestamp_ms}.json"
    if path.exists():
        return json.loads(path.read_bytes())
    raw = fetch(_url(timestamp_ms))
    payload = json.loads(raw)
    # A week whose last value is null is still being published; don't cache it.
    series = payload.get("series") or []
    if series and series[-1][1] is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return payload


def load_day_ahead_prices(
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: Path,
    fetch: Fetch = _http_get,
) -> pd.Series:
    """Hourly DE-LU day-ahead prices in EUR/MWh for delivery hours in [start, end).

    `start` and `end` must be timezone-aware and on full hours. The result has one
    entry per UTC hour in the range; hours SMARD has no complete data for are NaN.
    Weekly files are cached under `cache_dir / "smard"`.
    """
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start and end must be timezone-aware")
    # Check alignment in UTC: flooring local wall-clock time fails on the ambiguous autumn hour.
    start, end = start.tz_convert("UTC"), end.tz_convert("UTC")
    if start != start.floor("h") or end != end.floor("h"):
        raise ValueError("start and end must fall on full hours")
    if end <= start:
        raise ValueError("end must be after start")

    timestamps = sorted(json.loads(fetch(_url()))["timestamps"])
    start_ms, end_ms = start.value // 1_000_000, end.value // 1_000_000
    # Keep every weekly file that can overlap [start, end): it starts before `end`,
    # and the next file starts after `start`.
    chunk_starts = [
        t
        for t, next_t in zip(timestamps, [*timestamps[1:], None], strict=True)
        if t < end_ms and (next_t is None or next_t > start_ms)
    ]

    hourly_index = pd.date_range(start, end, freq="h", inclusive="left", name="delivery_start_utc")
    if not chunk_starts:
        return pd.Series(np.nan, index=hourly_index, name="price_eur_mwh")
    quarter_hours = pd.concat(
        parse_chart_data(_load_chunk(t, cache_dir, fetch)) for t in chunk_starts
    )
    if quarter_hours.index.has_duplicates:
        raise ValueError("SMARD weekly files overlap; refusing to pick between duplicate values")
    return quarter_hours_to_hourly(quarter_hours).reindex(hourly_index)
