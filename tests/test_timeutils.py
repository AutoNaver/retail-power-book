import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from rpb.timeutils import is_utc


@pytest.mark.parametrize("tz", ["UTC", "Etc/UTC", "GMT", ZoneInfo("Etc/UTC"), dt.UTC])
def test_utc_aliases_are_accepted(tz: object) -> None:
    assert is_utc(pd.date_range("2025-03-29", periods=72, freq="h", tz=tz))


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Etc/GMT-1"])
def test_naive_and_non_utc_are_rejected(tz: str | None) -> None:
    assert not is_utc(pd.date_range("2025-03-29", periods=72, freq="h", tz=tz))


def test_zone_with_utc_offset_only_part_of_the_time_is_rejected() -> None:
    # London is UTC+0 in winter but UTC+1 from 2025-03-30 01:00 UTC.
    assert not is_utc(pd.date_range("2025-03-29", periods=72, freq="h", tz="Europe/London"))
