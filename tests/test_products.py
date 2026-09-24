from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import pytest

from rpb.products import Product, delivery_index, product_mask


@pytest.mark.parametrize(
    ("day", "hours"),
    [(date(2024, 3, 31), 23), (date(2024, 10, 27), 25), (date(2024, 6, 12), 24)],
)
def test_day_hour_counts_follow_dst(day: date, hours: int) -> None:
    next_day = date.fromordinal(day.toordinal() + 1)
    assert len(delivery_index(day, next_day)) == hours


def test_index_is_utc_and_starts_at_local_midnight() -> None:
    idx = delivery_index(date(2024, 7, 1), date(2024, 7, 2))
    assert str(idx.tz) == "UTC"
    assert idx[0] == pd.Timestamp("2024-06-30 22:00", tz="UTC")


@pytest.mark.parametrize(
    ("product", "hours"),
    [
        (Product("base", 2024, "M", 3), 743),
        (Product("base", 2024, "M", 10), 745),
        (Product("base", 2025, "M", 1), 744),
        (Product("peak", 2025, "M", 1), 23 * 12),
        (Product("peak", 2024, "M", 3), 21 * 12),
    ],
)
def test_month_hour_counts(product: Product, hours: int) -> None:
    idx = delivery_index(product.start, product.end)
    assert product_mask(product, idx).sum() == hours


def test_calendar_year_2024_hours() -> None:
    assert len(delivery_index(date(2024, 1, 1), date(2025, 1, 1))) == 8784


def test_peak_hours_are_weekday_daytime_local() -> None:
    product = Product("peak", 2024, "M", 3)
    idx = delivery_index(date(2024, 3, 1), date(2024, 4, 1))
    local = idx[product_mask(product, idx)].tz_convert("Europe/Berlin")
    assert (local.dayofweek < 5).all()
    assert ((local.hour >= 8) & (local.hour < 20)).all()


def test_peak_week_across_dst_change_keeps_12_hours_per_weekday() -> None:
    # DST starts on Sunday 2024-03-31; the following weekdays keep 08:00-20:00 local.
    idx = delivery_index(date(2024, 3, 25), date(2024, 4, 8))
    mask = product_mask(Product("peak", 2024, "M", 4), idx)
    local = idx[mask].tz_convert("Europe/Berlin")
    per_day = pd.Series(1, index=local).groupby(local.date).sum()
    assert (per_day == 12).all()
    assert len(per_day) == 5  # 1-5 April 2024


@pytest.mark.parametrize("kind", ["base", "peak"])
def test_quarter_mask_is_union_of_month_masks(kind: str) -> None:
    quarter = Product(kind, 2024, "Q", 1)
    idx = delivery_index(date(2023, 12, 1), date(2024, 5, 1))
    union = np.logical_or.reduce([product_mask(m, idx) for m in quarter.months()])
    np.testing.assert_array_equal(product_mask(quarter, idx), union)


def test_mask_outside_index_period_is_empty() -> None:
    idx = delivery_index(date(2024, 1, 1), date(2024, 2, 1))
    assert not product_mask(Product("base", 2024, "M", 2), idx).any()


def test_names_and_periods() -> None:
    q4 = Product("peak", 2025, "Q", 4)
    assert q4.name == "DE-PEAK-2025-Q4"
    assert Product("base", 2025, "M", 1).name == "DE-BASE-2025-M01"
    assert (q4.start, q4.end) == (date(2025, 10, 1), date(2026, 1, 1))


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        Product("base", 2025, "Q", 5)
    with pytest.raises(ValueError):
        Product("offpeak", 2025, "M", 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        delivery_index(datetime(2024, 1, 1, 6, tzinfo=UTC), date(2024, 1, 2))
    with pytest.raises(ValueError):
        delivery_index(date(2024, 1, 2), date(2024, 1, 1))
    naive = pd.date_range("2024-01-01", periods=24, freq="h")
    with pytest.raises(ValueError):
        product_mask(Product("base", 2024, "M", 1), naive)


def test_mask_accepts_utc_aliases() -> None:
    idx = delivery_index(date(2024, 3, 1), date(2024, 4, 1))
    product = Product("peak", 2024, "M", 3)
    expected = product_mask(product, idx)
    for alias in ("Etc/UTC", "GMT"):
        np.testing.assert_array_equal(product_mask(product, idx.tz_convert(alias)), expected)


def test_mask_rejects_local_time_index() -> None:
    idx = delivery_index(date(2024, 3, 1), date(2024, 4, 1)).tz_convert("Europe/Berlin")
    with pytest.raises(ValueError, match="UTC"):
        product_mask(Product("base", 2024, "M", 3), idx)
