from datetime import date

import numpy as np
import pandas as pd
import pytest

from rpb.models.weather import (
    TemperatureNormal,
    fit_temperature_normal,
    german_temperature_anomaly,
    map_weather_year,
    temperature_anomaly,
    temperature_normal,
    weather_hours,
)
from rpb.products import delivery_index


def known_normal(level: float = 9.0) -> TemperatureNormal:
    """A normal inside the model class: level, trend, 3 harmonics, varying by hour of day."""
    hours = np.arange(24)
    diurnal = 4.0 * np.sin(2 * np.pi * (hours - 9) / 24)
    coefficients = np.column_stack(
        [
            level + diurnal,  # level
            np.full(24, 0.04),  # trend, °C per year
            np.full(24, -8.0),  # cos1: cold in January
            np.full(24, -2.0) + 0.05 * hours,  # sin1
            np.full(24, 0.5),  # cos2
            np.full(24, -0.3),  # sin2
            np.full(24, 0.1),  # cos3
            np.full(24, 0.2),  # sin3
        ]
    )
    return TemperatureNormal(coefficients=coefficients, n_harmonics=3)


def history(model: TemperatureNormal, noise_c: float = 0.0, seed: int = 0) -> pd.Series:
    index = pd.date_range("2006-01-01", "2025-01-01", freq="h", inclusive="left", tz="UTC")
    values = temperature_normal(model, index)
    if noise_c:
        values = values + np.random.default_rng(seed).normal(0.0, noise_c, size=len(index))
    return pd.Series(values, index=index, name="temperature_c")


def test_fit_recovers_known_normal_exactly_without_noise() -> None:
    model = known_normal()
    fitted = fit_temperature_normal(history(model))
    np.testing.assert_allclose(fitted.coefficients, model.coefficients, atol=1e-8)


def test_fit_recovers_trend_and_seasonality_with_noise() -> None:
    model = known_normal()
    fitted = fit_temperature_normal(history(model, noise_c=3.0, seed=7))
    # 19 years x 365 days per hour of day: the trend's standard error is about 0.004 °C/year.
    np.testing.assert_allclose(fitted.coefficients[:, 1], 0.04, atol=0.02)
    probe = delivery_index(date(2026, 1, 1), date(2027, 1, 1))
    np.testing.assert_allclose(
        temperature_normal(fitted, probe), temperature_normal(model, probe), atol=0.5
    )


def test_anomaly_of_normal_weather_is_zero() -> None:
    model = known_normal()
    series = history(model)
    anomaly = temperature_anomaly(series, model)
    np.testing.assert_allclose(anomaly.to_numpy(), 0.0, atol=1e-9)
    assert anomaly.name == "temperature_anomaly_c"


def test_fit_drops_nan_and_rejects_inf() -> None:
    model = known_normal()
    series = history(model)
    gappy = series.copy()
    gappy.iloc[::13] = np.nan
    np.testing.assert_allclose(
        fit_temperature_normal(gappy).coefficients, model.coefficients, atol=1e-8
    )
    gappy.iloc[5] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        fit_temperature_normal(gappy)


def test_fit_needs_two_years() -> None:
    index = pd.date_range("2024-01-01", periods=24 * 300, freq="h", tz="UTC")
    with pytest.raises(ValueError, match="not enough observations"):
        fit_temperature_normal(pd.Series(10.0, index=index))


def test_missing_cold_station_does_not_create_an_anomaly() -> None:
    warm, cold = known_normal(level=12.0), known_normal(level=2.0)
    index = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    temps = {
        "00001": pd.Series(temperature_normal(warm, index), index=index),
        "00002": pd.Series(temperature_normal(cold, index), index=index),
    }
    temps["00002"].iloc[:24] = np.nan  # the cold station is missing on day 1
    anomaly = german_temperature_anomaly(
        temps, {"00001": warm, "00002": cold}, {"00001": 0.8, "00002": 0.2}, 0.8
    )
    np.testing.assert_allclose(anomaly.to_numpy(), 0.0, atol=1e-9)
    assert anomaly.name == "temperature_anomaly_c"


def test_non_leap_years_map_to_the_same_utc_time() -> None:
    index = delivery_index(date(2025, 1, 1), date(2026, 1, 1))
    hours = weather_hours(index, 2019)
    assert len(hours) == len(index) == 8760
    expected = pd.DatetimeIndex([t.replace(year=t.year - 6) for t in index], tz="UTC")
    assert hours.equals(expected.rename("weather_utc"))


def test_autumn_dst_day_maps_to_distinct_consecutive_hours() -> None:
    index = delivery_index(date(2025, 10, 26), date(2025, 10, 27))
    hours = weather_hours(index, 2019)
    assert len(hours) == 25
    assert ((hours[1:] - hours[:-1]) == pd.Timedelta(hours=1)).all()


def test_leap_day_maps_to_28_february() -> None:
    index = delivery_index(date(2024, 2, 28), date(2024, 3, 2))
    hours = weather_hours(index, 2019)
    std = hours.tz_convert("UTC").tz_localize(None) + pd.Timedelta(hours=1)
    local_std = index.tz_localize(None) + pd.Timedelta(hours=1)
    leap = (local_std.month == 2) & (local_std.day == 29)
    assert leap.sum() == 24
    assert set(std[leap].day) == {28} and set(std[leap].month) == {2}
    assert set(std[leap].year) == {2019}
    # 28 February appears twice: once for 28 Feb and once for 29 Feb.
    assert hours.has_duplicates and len(hours.unique()) == len(hours) - 24


def test_weather_leap_day_unused_for_non_leap_delivery() -> None:
    index = delivery_index(date(2025, 1, 1), date(2026, 1, 1))
    std = weather_hours(index, 2024).tz_localize(None) + pd.Timedelta(hours=1)
    assert not ((std.month == 2) & (std.day == 29)).any()
    assert len(std) == 8760 and not std.has_duplicates


def test_multi_year_delivery_maps_to_consecutive_weather_years() -> None:
    index = delivery_index(date(2025, 12, 31), date(2026, 1, 2))
    std = weather_hours(index, 2010).tz_localize(None) + pd.Timedelta(hours=1)
    assert list(np.unique(std.year)) == [2010, 2011]
    assert (
        (std[1:] - std[:-1]) == pd.Timedelta(hours=1)
    ).all()  # stays consecutive across new year


def test_map_weather_year_takes_values_and_keeps_gaps() -> None:
    weather_index = pd.date_range("2019-01-01", "2020-01-01", freq="h", inclusive="left", tz="UTC")
    series = pd.Series(np.arange(len(weather_index), dtype=float), index=weather_index)
    series.iloc[100] = np.nan
    index = pd.date_range("2025-01-01", periods=200, freq="h", tz="UTC")
    values = map_weather_year(series, index, 2019)
    np.testing.assert_array_equal(np.isnan(values), np.arange(200) == 100)
    assert values[0] == 0.0 and values[199] == 199.0


def test_weather_hours_rejects_non_utc() -> None:
    with pytest.raises(ValueError, match="UTC"):
        weather_hours(pd.date_range("2025-01-01", periods=3, freq="h", tz="Europe/Berlin"), 2019)
