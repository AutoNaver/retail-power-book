from datetime import date

import numpy as np
import pandas as pd
import pytest

from rpb.models.load import deterministic_load_mwh, profile_shape
from rpb.products import delivery_index


@pytest.mark.parametrize("year", [2024, 2025])
def test_shape_sums_to_one_over_a_local_year(year: int) -> None:
    index = delivery_index(date(year, 1, 1), date(year + 1, 1, 1))
    shape = profile_shape(index)
    assert len(shape) == (8784 if year == 2024 else 8760)
    assert shape.sum() == pytest.approx(1.0, abs=1e-12)
    assert (shape > 0).all()


def test_sub_range_is_its_share_of_the_year() -> None:
    year = profile_shape(delivery_index(date(2025, 1, 1), date(2026, 1, 1)))
    january = profile_shape(delivery_index(date(2025, 1, 1), date(2025, 2, 1)))
    np.testing.assert_allclose(january, year[:744], rtol=1e-12)
    assert 0.05 < january.sum() < 0.15  # a winter month, above 1/12 but not wildly


def test_range_across_years_normalises_each_year() -> None:
    index = delivery_index(date(2024, 12, 1), date(2025, 2, 1))
    shape = pd.Series(profile_shape(index), index=index.tz_convert("Europe/Berlin"))
    dec = profile_shape(delivery_index(date(2024, 12, 1), date(2025, 1, 1)))
    np.testing.assert_allclose(shape[shape.index.year == 2024].to_numpy(), dec, rtol=1e-12)


@pytest.mark.parametrize(("day", "hours"), [(date(2025, 3, 30), 23), (date(2025, 10, 26), 25)])
def test_dst_days_have_23_and_25_hours(day: date, hours: int) -> None:
    next_day = date.fromordinal(day.toordinal() + 1)
    assert len(profile_shape(delivery_index(day, next_day))) == hours


def test_holiday_takes_the_sunday_curve() -> None:
    def daily_curve(day: date) -> np.ndarray:
        values = profile_shape(delivery_index(day, date.fromordinal(day.toordinal() + 1)))
        return values / values.sum()

    may_day, sunday, thursday = date(2025, 5, 1), date(2025, 5, 4), date(2025, 5, 8)
    # Same month: curves differ only by the daily dynamisation factor, removed by normalising.
    np.testing.assert_allclose(daily_curve(may_day), daily_curve(sunday), rtol=1e-9)
    assert not np.allclose(daily_curve(may_day), daily_curve(thursday), rtol=1e-3)


def test_deterministic_load_known_result() -> None:
    index = delivery_index(date(2025, 1, 1), date(2026, 1, 1))
    load = deterministic_load_mwh(index, n_customers=10_000, annual_mwh_per_customer=3.5)
    np.testing.assert_array_equal(load, 10_000 * 3.5 * profile_shape(index))
    assert load.sum() == pytest.approx(35_000.0, rel=1e-12)


def test_invalid_inputs_raise() -> None:
    index = delivery_index(date(2025, 1, 1), date(2025, 1, 2))
    with pytest.raises(ValueError):
        deterministic_load_mwh(index, n_customers=-1, annual_mwh_per_customer=3.5)
    with pytest.raises(ValueError, match="UTC"):
        profile_shape(index.tz_convert("Europe/Berlin"))
    with pytest.raises(ValueError, match="whole UTC hours"):
        profile_shape(index + pd.Timedelta(minutes=30))


@pytest.mark.parametrize("annual", [np.nan, np.inf, -np.inf, -1.0])
def test_non_finite_or_negative_consumption_raises(annual: float) -> None:
    index = delivery_index(date(2025, 1, 1), date(2025, 1, 2))
    with pytest.raises(ValueError, match="finite and non-negative"):
        deterministic_load_mwh(index, n_customers=100, annual_mwh_per_customer=annual)


@pytest.mark.parametrize("customers", [2.5, True, np.nan])
def test_non_integer_customer_count_raises(customers: object) -> None:
    index = delivery_index(date(2025, 1, 1), date(2025, 1, 2))
    with pytest.raises(TypeError, match="integer"):
        deterministic_load_mwh(index, n_customers=customers, annual_mwh_per_customer=3.5)  # type: ignore[arg-type]
