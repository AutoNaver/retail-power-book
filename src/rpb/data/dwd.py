"""Hourly air temperature from DWD station observations.

Data source: Deutscher Wetterdienst (DWD), Climate Data Center, licensed under
CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).

DWD serves one zip per station in two directories:

    {BASE_URL}/historical/stundenwerte_TU_{id}_{from}_{to}_hist.zip   quality-checked, versioned
    {BASE_URL}/recent/stundenwerte_TU_{id}_akt.zip                    roughly the last 500 days

Each zip holds `produkt_tu_stunde_*.txt` (semicolon-separated; MESS_DATUM is
YYYYMMDDHH, TT_TU is air temperature in °C, -999 is missing) and
`Metadaten_Parameter_tu_stunde_*.txt`, which says per period whether timestamps
are in UTC or in MEZ (CET, fixed UTC+1 with no DST). Older periods are often MEZ.
"""

import io
import re
import urllib.request
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from rpb.timeutils import require_utc

BASE_URL = (
    "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/"
    "air_temperature"
)
MISSING = -999.0
UTC_OFFSET_HOURS = {"UTC": 0, "MEZ": 1}
ENCODING = "latin-1"

Fetch = Callable[[str], bytes]


def _http_get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "retail-power-book"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _member(archive: zipfile.ZipFile, prefix: str) -> str:
    # Metadata comes as both .txt and .html; the .txt is the machine-readable one.
    names = [n for n in archive.namelist() if n.startswith(prefix) and n.endswith(".txt")]
    if len(names) != 1:
        raise ValueError(f"expected one '{prefix}*.txt' file in the DWD zip, found {names}")
    return names[0]


def _time_basis_periods(metadata: str) -> list[tuple[int, int, str]]:
    """(from, to, "UTC" | "MEZ") periods for TT_TU from Metadaten_Parameter, dates as YYYYMMDD."""
    periods = []
    for line in metadata.splitlines():
        fields = [f.strip() for f in line.split(";")]
        if len(fields) < 9 or fields[4] != "TT_TU":
            continue
        match = re.search(r"Stundenwerte in (\w+)", line)
        if match is None or match.group(1) not in UTC_OFFSET_HOURS:
            raise ValueError(f"unknown time basis in DWD metadata line: {line!r}")
        periods.append((int(fields[1]), int(fields[2]), match.group(1)))
    if not periods:
        raise ValueError("DWD metadata has no TT_TU periods")
    return periods


def parse_station_zip(
    content: bytes, first_day: int | None = None, last_day: int | None = None
) -> pd.Series:
    """Hourly temperature in °C from one DWD station zip, on a UTC index.

    Only rows whose MESS_DATUM day (YYYYMMDD) is within [first_day, last_day] are
    kept, when given. Timestamps are converted to UTC using the time basis the
    metadata gives for each period. -999 becomes NaN. A kept row whose day falls in
    no period, or in two periods with different time bases, raises: some stations
    switch basis on a day that both periods claim, and those hours can't be placed.
    """
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        product = archive.read(_member(archive, "produkt_tu_stunde_")).decode(ENCODING)
        metadata = archive.read(_member(archive, "Metadaten_Parameter_")).decode(ENCODING)

    table = pd.read_csv(
        io.StringIO(product), sep=";", skipinitialspace=True, dtype={"MESS_DATUM": str}
    )
    table.columns = [c.strip() for c in table.columns]
    all_days = table["MESS_DATUM"].str[:8].astype(int)
    keep = np.ones(len(table), dtype=bool)
    if first_day is not None:
        keep &= all_days.to_numpy() >= first_day
    if last_day is not None:
        keep &= all_days.to_numpy() <= last_day
    table = table[keep].reset_index(drop=True)
    stamps = pd.to_datetime(table["MESS_DATUM"], format="%Y%m%d%H")
    days = table["MESS_DATUM"].str[:8].astype(int).to_numpy()

    offset_hours = np.full(len(table), np.nan)
    for start, end, basis in _time_basis_periods(metadata):
        in_period = (days >= start) & (days <= end)
        conflict = in_period & ~np.isnan(offset_hours) & (offset_hours != UTC_OFFSET_HOURS[basis])
        if conflict.any():
            raise ValueError(f"DWD metadata gives conflicting time bases on {days[conflict][0]}")
        offset_hours[in_period] = UTC_OFFSET_HOURS[basis]
    if np.isnan(offset_hours).any():
        first = table["MESS_DATUM"][np.isnan(offset_hours)].iloc[0]
        raise ValueError(f"no time basis in DWD metadata for MESS_DATUM {first}")

    utc = (stamps - pd.to_timedelta(offset_hours, unit="h")).dt.tz_localize("UTC")
    values = table["TT_TU"].astype(float).where(table["TT_TU"] != MISSING)
    series = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(utc), name="temperature_c")
    series.index.name = "observation_utc"
    if series.index.has_duplicates:
        raise ValueError("DWD station file has duplicate timestamps after UTC conversion")
    return series.sort_index()


def _historical_name(station_id: str, listing: str) -> str:
    names = set(re.findall(rf"stundenwerte_TU_{station_id}_\d{{8}}_\d{{8}}_hist\.zip", listing))
    if len(names) != 1:
        raise ValueError(f"expected one historical file for station {station_id}, found {names}")
    return names.pop()


def load_station_temperature(
    station_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: Path,
    fetch: Fetch = _http_get,
) -> pd.Series:
    """Hourly temperature in °C at one DWD station for UTC hours in [start, end).

    Historical (quality-checked) data is used up to the last day in its file name;
    recent data only after that day. Hours without an observation are NaN.
    The historical zip is cached under `cache_dir / "dwd"` (its file name changes
    when DWD publishes a new version); the recent zip changes daily and is not cached.
    """
    if not re.fullmatch(r"\d{5}", station_id):
        raise ValueError(f"DWD station id must be 5 digits, got {station_id!r}")
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start and end must be timezone-aware")
    start, end = start.tz_convert("UTC"), end.tz_convert("UTC")
    if start != start.floor("h") or end != end.floor("h") or end <= start:
        raise ValueError("start and end must be full hours with end after start")

    # Local station days can differ from UTC days by at most one.
    first_day = int((start - pd.Timedelta(days=1)).strftime("%Y%m%d"))
    last_day = int((end + pd.Timedelta(days=1)).strftime("%Y%m%d"))

    name = _historical_name(station_id, fetch(f"{BASE_URL}/historical/").decode())
    path = cache_dir / "dwd" / name
    if path.exists():
        historical_zip = path.read_bytes()
    else:
        historical_zip = fetch(f"{BASE_URL}/historical/{name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(historical_zip)
    series = parse_station_zip(historical_zip, first_day, last_day)

    # The historical file name ends with its last day; recent data is used only after it.
    historical_last_day = int(name.split("_")[-2])
    if last_day > historical_last_day:
        recent_zip = fetch(f"{BASE_URL}/recent/stundenwerte_TU_{station_id}_akt.zip")
        recent = parse_station_zip(recent_zip, max(first_day, historical_last_day + 1), last_day)
        series = pd.concat([series, recent])

    hours = pd.date_range(start, end, freq="h", inclusive="left", name="observation_utc")
    return series.reindex(hours).rename(f"temperature_c_{station_id}")


def weighted_temperature(
    station_temperatures: Mapping[str, pd.Series],
    weights: Mapping[str, float],
    min_reporting_weight: float,
) -> pd.Series:
    """Weighted average temperature in °C across stations, per UTC hour.

    When some stations are missing an hour, the average uses the stations that
    reported, with their weights rescaled to sum to 1, provided they carry at least
    `min_reporting_weight` of the total weight. Otherwise the hour is NaN.
    """
    if set(station_temperatures) != set(weights):
        raise ValueError("stations and weights must have the same station ids")
    if not 0.0 < min_reporting_weight <= 1.0:
        raise ValueError(f"min_reporting_weight must be in (0, 1], got {min_reporting_weight}")
    for station_id, series in station_temperatures.items():
        try:
            require_utc(series.index)
        except ValueError as error:
            raise ValueError(f"station {station_id}: {error}") from error
        if np.isinf(series.to_numpy(dtype=float)).any():
            raise ValueError(f"station {station_id}: temperatures must be finite or NaN")

    frame = pd.DataFrame(dict(station_temperatures))
    w = pd.Series(weights, dtype=float)[frame.columns].to_numpy()
    if (w <= 0).any():
        raise ValueError("station weights must be > 0")
    w = w / w.sum()

    values = frame.to_numpy(dtype=float)
    reporting = ~np.isnan(values)
    reporting_weight = reporting @ w
    weighted_sum = np.where(reporting, values, 0.0) @ w
    enough = reporting_weight >= min_reporting_weight - 1e-12
    result = np.full(len(frame), np.nan)
    result[enough] = weighted_sum[enough] / reporting_weight[enough]
    return pd.Series(result, index=frame.index.tz_convert("UTC"), name="temperature_c")
