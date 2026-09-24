import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rpb.data.dwd import (
    BASE_URL,
    load_station_temperature,
    parse_station_zip,
    weighted_temperature,
)

FIXTURES = Path(__file__).parent / "fixtures" / "dwd"
# Trimmed from the real DWD files for Berlin-Tempelhof (00433); see the PR for the windows kept.
HIST_NAME = "stundenwerte_TU_00433_19510101_20251231_hist.zip"
HIST_ZIP = (FIXTURES / HIST_NAME).read_bytes()
RECENT_ZIP = (FIXTURES / "stundenwerte_TU_00433_akt.zip").read_bytes()
LISTING = f'<a href="{HIST_NAME}">{HIST_NAME}</a> <a href="stundenwerte_TU_00044_20070401_20251231_hist.zip">'


def utc(ts: str) -> pd.Timestamp:
    return pd.Timestamp(ts, tz="UTC")


def rewrite_product(content: bytes, edit: dict[str, str]) -> bytes:
    """Copy of a station zip with some TT_TU values replaced, keyed by MESS_DATUM."""
    source = zipfile.ZipFile(io.BytesIO(content))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as target:
        for name in source.namelist():
            data = source.read(name)
            if name.startswith("produkt_"):
                lines = data.decode("latin-1").split("\n")
                for i, line in enumerate(lines):
                    fields = line.split(";")
                    if len(fields) > 3 and fields[1].strip() in edit:
                        fields[3] = edit[fields[1].strip()]
                        lines[i] = ";".join(fields)
                data = "\n".join(lines).encode("latin-1")
            target.writestr(name, data)
    return out.getvalue()


class FakeDwd:
    def __init__(self, recent: bytes = RECENT_ZIP) -> None:
        self.recent = recent
        self.urls: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.urls.append(url)
        responses = {
            f"{BASE_URL}/historical/": LISTING.encode(),
            f"{BASE_URL}/historical/{HIST_NAME}": HIST_ZIP,
            f"{BASE_URL}/recent/stundenwerte_TU_00433_akt.zip": self.recent,
        }
        if url not in responses:
            raise AssertionError(f"unexpected URL {url}")
        return responses[url]


def test_cet_period_is_converted_to_utc() -> None:
    series = parse_station_zip(HIST_ZIP)
    # Until 1995-01-31 Tempelhof reports in MEZ (UTC+1); from 1995-02-01 in UTC.
    assert series[utc("1995-01-31 21:00")] == pytest.approx(0.9)  # 22 MEZ
    assert series[utc("1995-01-31 22:00")] == pytest.approx(1.8)  # 23 MEZ
    assert utc("1995-01-31 23:00") not in series.index  # the switch leaves a one-hour gap
    assert series[utc("1995-02-01 00:00")] == pytest.approx(2.2)
    assert str(series.index.tz) == "UTC"
    assert series.name == "temperature_c"


def test_missing_values_become_nan() -> None:
    series = parse_station_zip(HIST_ZIP)
    assert series[utc("2024-04-23 06:00")] == pytest.approx(1.9)
    assert series[utc("2024-04-23 07:00") : utc("2024-04-23 10:00")].isna().all()


def test_load_gives_24_utc_hours_per_day_across_dst(tmp_path: Path) -> None:
    temps = load_station_temperature(
        "00433", utc("2024-03-30 00:00"), utc("2024-04-01 00:00"), tmp_path, fetch=FakeDwd()
    )
    assert len(temps) == 48
    assert temps.notna().all()
    assert temps.name == "temperature_c_00433"


def test_load_marks_the_switch_gap_as_nan(tmp_path: Path) -> None:
    temps = load_station_temperature(
        "00433", utc("1995-01-31 20:00"), utc("1995-02-01 02:00"), tmp_path, fetch=FakeDwd()
    )
    assert temps.isna().sum() == 1
    assert np.isnan(temps[utc("1995-01-31 23:00")])


def test_recent_data_only_after_historical_ends(tmp_path: Path) -> None:
    # Make recent disagree with historical on the overlap, so we can tell which one is used.
    recent = rewrite_product(RECENT_ZIP, {"2025123122": "99.9", "2025123123": "99.9"})
    temps = load_station_temperature(
        "00433", utc("2025-12-31 22:00"), utc("2026-01-01 06:00"), tmp_path, fetch=FakeDwd(recent)
    )
    assert temps[utc("2025-12-31 22:00")] == pytest.approx(3.3)  # historical
    assert temps[utc("2025-12-31 23:00")] == pytest.approx(3.5)  # historical
    assert temps[utc("2026-01-01 00:00")] == pytest.approx(3.7)  # recent
    assert temps[utc("2026-01-01 05:00")] == pytest.approx(2.7)
    assert temps.notna().all()


def test_recent_not_fetched_when_historical_covers_the_range(tmp_path: Path) -> None:
    fetch = FakeDwd()
    load_station_temperature(
        "00433", utc("2024-03-30 00:00"), utc("2024-03-31 00:00"), tmp_path, fetch=fetch
    )
    assert not any("/recent/" in url for url in fetch.urls)


def test_historical_zip_is_cached(tmp_path: Path) -> None:
    args = ("00433", utc("2024-03-30 00:00"), utc("2024-03-31 00:00"), tmp_path)
    load_station_temperature(*args, fetch=FakeDwd())
    assert (tmp_path / "dwd" / HIST_NAME).exists()
    second = FakeDwd()
    load_station_temperature(*args, fetch=second)
    assert second.urls == [f"{BASE_URL}/historical/"]  # only the listing


def rewrite_metadata(content: bytes, old: bytes, new: bytes) -> bytes:
    source = zipfile.ZipFile(io.BytesIO(content))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as target:
        for name in source.namelist():
            data = source.read(name)
            if name.startswith("Metadaten_Parameter") and name.endswith(".txt"):
                assert old in data
                data = data.replace(old, new)
            target.writestr(name, data)
    return out.getvalue()


def test_unknown_time_basis_raises() -> None:
    content = rewrite_metadata(HIST_ZIP, b"Stundenwerte in MEZ", b"Stundenwerte in MESZ")
    with pytest.raises(ValueError, match="time basis"):
        parse_station_zip(content)


def test_day_claimed_by_both_bases_raises_only_when_requested(tmp_path: Path) -> None:
    # Like Hannover (02014), whose MEZ period ends on the day its UTC period starts.
    content = rewrite_metadata(HIST_ZIP, b"19510101;19950131", b"19510101;19950201")
    with pytest.raises(ValueError, match="conflicting time bases on 19950201"):
        parse_station_zip(content)

    class Fetch(FakeDwd):
        def __call__(self, url: str) -> bytes:
            return content if url.endswith(HIST_NAME) else super().__call__(url)

    temps = load_station_temperature(
        "00433", utc("2024-03-30"), utc("2024-03-31"), tmp_path, fetch=Fetch()
    )
    assert temps.notna().all()
    with pytest.raises(ValueError, match="conflicting"):
        load_station_temperature(
            "00433", utc("1995-01-31 20:00"), utc("1995-02-01 02:00"), tmp_path / "x", fetch=Fetch()
        )


def test_real_zip_layout_with_html_metadata_is_parsed() -> None:
    names = zipfile.ZipFile(io.BytesIO(HIST_ZIP)).namelist()
    assert any(n.endswith(".html") for n in names)  # as in the real DWD zips
    assert not parse_station_zip(HIST_ZIP).empty


@pytest.mark.parametrize(
    ("station", "start", "end"),
    [
        ("433", utc("2024-03-30"), utc("2024-03-31")),
        ("00433", pd.Timestamp("2024-03-30"), utc("2024-03-31")),
        ("00433", utc("2024-03-30 00:30"), utc("2024-03-31")),
        ("00433", utc("2024-03-31"), utc("2024-03-30")),
    ],
)
def test_invalid_arguments_raise(
    station: str, start: pd.Timestamp, end: pd.Timestamp, tmp_path: Path
) -> None:
    with pytest.raises(ValueError):
        load_station_temperature(station, start, end, tmp_path, fetch=FakeDwd())


def test_weighted_temperature_known_result() -> None:
    index = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    temps = {
        "00433": pd.Series([0.0, 4.0, 8.0], index=index),
        "01975": pd.Series([4.0, 8.0, np.nan], index=index),
    }
    result = weighted_temperature(temps, {"00433": 0.25, "01975": 0.75})
    np.testing.assert_allclose(result.to_numpy()[:2], [3.0, 7.0])
    assert np.isnan(result.iloc[2])  # a missing station is not silently dropped
    assert result.name == "temperature_c"


def test_weighted_temperature_needs_matching_stations() -> None:
    index = pd.date_range("2025-01-01", periods=1, freq="h", tz="UTC")
    with pytest.raises(ValueError, match="same station ids"):
        weighted_temperature({"00433": pd.Series([1.0], index=index)}, {"01975": 1.0})
