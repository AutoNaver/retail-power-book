from datetime import date

import numpy as np
import pandas as pd
import pytest

from rpb.holidays import day_type, easter_sunday, german_public_holidays
from rpb.products import delivery_index


@pytest.mark.parametrize(
    "easter",
    [
        date(2019, 4, 21),
        date(2024, 3, 31),
        date(2025, 4, 20),
        date(2026, 4, 5),
        date(2285, 3, 22),  # earliest possible date
        date(2038, 4, 25),  # latest possible date
    ],
)
def test_easter_sunday_known_dates(easter: date) -> None:
    assert easter_sunday(easter.year) == easter


def test_holidays_2025() -> None:
    assert german_public_holidays(2025) == {
        date(2025, 1, 1),
        date(2025, 4, 18),
        date(2025, 4, 21),
        date(2025, 5, 1),
        date(2025, 5, 29),
        date(2025, 6, 9),
        date(2025, 10, 3),
        date(2025, 12, 25),
        date(2025, 12, 26),
    }


def test_reformation_day_only_in_2017() -> None:
    assert len(german_public_holidays(2017)) == 10
    assert date(2017, 10, 31) in german_public_holidays(2017)
    assert date(2018, 10, 31) not in german_public_holidays(2018)


def test_day_type_uses_local_date() -> None:
    index = pd.DatetimeIndex(["2024-12-31 22:00", "2024-12-31 23:00"], tz="UTC")
    # 23:00 and 00:00 local: New Year's Eve is a Tuesday, New Year's Day a holiday.
    assert list(day_type(index)) == ["weekday", "sunday_holiday"]


def test_week_day_types() -> None:
    # Monday 2025-04-14 to Monday 2025-04-21: Good Friday 18th, Easter Monday 21st.
    index = delivery_index(date(2025, 4, 14), date(2025, 4, 22))
    daily = pd.Series(day_type(index), index=index.tz_convert("Europe/Berlin").date)
    per_day = daily.groupby(level=0).agg(lambda s: set(s))
    assert [next(iter(v)) for v in per_day] == [
        "weekday",
        "weekday",
        "weekday",
        "weekday",
        "sunday_holiday",  # Good Friday
        "saturday",
        "sunday_holiday",
        "sunday_holiday",  # Easter Monday
    ]
    assert all(len(v) == 1 for v in per_day)


def test_holiday_on_saturday_is_sunday_holiday() -> None:
    # 2026-12-26 is a Saturday.
    index = delivery_index(date(2026, 12, 26), date(2026, 12, 27))
    assert (day_type(index) == "sunday_holiday").all()


def test_dst_sunday_has_25_sunday_holiday_hours() -> None:
    index = delivery_index(date(2024, 10, 27), date(2024, 10, 28))
    types = day_type(index)
    assert len(types) == 25
    assert (types == "sunday_holiday").all()


def test_day_type_rejects_local_index() -> None:
    index = pd.date_range("2025-01-01", periods=3, freq="h", tz="Europe/Berlin")
    with pytest.raises(ValueError, match="UTC"):
        day_type(index)


def test_day_type_values_are_known_labels() -> None:
    index = delivery_index(date(2024, 1, 1), date(2025, 1, 1))
    assert set(np.unique(day_type(index))) == {"weekday", "saturday", "sunday_holiday"}
