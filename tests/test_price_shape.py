from datetime import date

import numpy as np
import pandas as pd
import pytest

from rpb.holidays import DAY_TYPES
from rpb.models.price import calendar_keys, estimate_shape, evaluate_shape
from rpb.products import delivery_index

AMPLITUDE = {"weekday": 30.0, "saturday": 20.0, "sunday_holiday": 12.0}


def true_shape(month: np.ndarray, day_type: np.ndarray, hour: np.ndarray) -> np.ndarray:
    """A known shape with zero mean over the 24 hours of every (month, day type)."""
    amplitude = np.vectorize(AMPLITUDE.get)(day_type) * (1.0 + 0.05 * month)
    return amplitude * np.sin(2 * np.pi * (hour - 6) / 24) + 5.0 * np.cos(4 * np.pi * hour / 24)


def synthetic_prices(seed: int, noise_eur_mwh: float = 1.0) -> tuple[pd.Series, dict]:
    index = delivery_index(date(2021, 1, 1), date(2025, 1, 1))
    keys = calendar_keys(index)
    rng = np.random.default_rng(seed)
    # A random level per local (year, month), including negative levels.
    year_month = keys["year"] * 12 + keys["month"]
    months, inverse = np.unique(year_month, return_inverse=True)
    level = rng.uniform(-50.0, 250.0, size=len(months))[inverse]
    shape = true_shape(keys["month"], keys["day_type"], keys["hour"])
    noise = rng.normal(0.0, noise_eur_mwh, size=len(index))
    return pd.Series(level + shape + noise, index=index, name="price_eur_mwh"), keys


def test_recovers_known_shape_whatever_the_levels() -> None:
    prices, _ = synthetic_prices(seed=11)
    estimate = estimate_shape(prices)

    assert estimate.notna().all()
    cells = estimate.index
    expected = true_shape(
        cells.get_level_values("month").to_numpy(),
        cells.get_level_values("day_type").to_numpy(),
        cells.get_level_values("hour").to_numpy(),
    )
    # Noise sigma 1 over >= ~16 days per cell, plus a small DST-day distortion.
    np.testing.assert_allclose(estimate.to_numpy(), expected, atol=1.5)


def test_shape_is_invariant_to_a_price_shift() -> None:
    prices, _ = synthetic_prices(seed=12)
    pd.testing.assert_series_equal(estimate_shape(prices), estimate_shape(prices - 300.0))


def test_constant_price_gives_zero_shape() -> None:
    index = delivery_index(date(2024, 1, 1), date(2025, 1, 1))
    shape = estimate_shape(pd.Series(-7.5, index=index))
    assert len(shape) == 12 * len(DAY_TYPES) * 24
    assert shape.notna().all()
    np.testing.assert_allclose(shape.to_numpy(), 0.0, atol=1e-12)


def test_evaluated_shape_has_zero_mean_in_every_local_month() -> None:
    prices, _ = synthetic_prices(seed=13)
    shape = estimate_shape(prices)
    index = delivery_index(date(2025, 1, 1), date(2026, 1, 1))  # includes both DST months
    values = evaluate_shape(shape, index)

    local = index.tz_convert("Europe/Berlin")
    monthly_mean = pd.Series(values).groupby(np.asarray(local.month)).mean()
    np.testing.assert_allclose(monthly_mean.to_numpy(), 0.0, atol=1e-9)
    assert len(values) == 8760
    assert np.std(values) > 10.0  # it is a real shape, not zeros


def test_nan_prices_are_dropped_not_propagated() -> None:
    prices, _ = synthetic_prices(seed=14)
    gappy = prices.copy()
    gappy.iloc[::97] = np.nan
    pd.testing.assert_series_equal(estimate_shape(gappy), estimate_shape(gappy.dropna()))
    assert estimate_shape(gappy).notna().all()


def test_missing_cell_raises_on_evaluation() -> None:
    index = delivery_index(date(2024, 1, 1), date(2024, 2, 1))
    shape = estimate_shape(pd.Series(50.0, index=index))  # January only
    assert shape.loc[2].isna().all()
    evaluate_shape(shape, index)  # January is fine
    with pytest.raises(ValueError, match="no value"):
        evaluate_shape(shape, delivery_index(date(2024, 2, 1), date(2024, 3, 1)))


def test_all_nan_prices_raise() -> None:
    index = delivery_index(date(2024, 1, 1), date(2024, 1, 2))
    with pytest.raises(ValueError, match="no non-NaN"):
        estimate_shape(pd.Series(np.nan, index=index))
