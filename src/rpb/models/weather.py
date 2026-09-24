"""Temperature normals, anomalies and weather-year mapping (docs/design.md section 4).

Everything here keys hours in standard time: UTC+1 all year (MEZ), with no DST, so
every day has 24 hours and the key follows the sun.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rpb.data.dwd import weighted_temperature
from rpb.timeutils import require_utc

STANDARD_TIME_OFFSET = pd.Timedelta(hours=1)
TREND_REFERENCE_YEAR = 2000.0


def standard_time(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Naive wall-clock times in UTC+1 for a UTC index."""
    require_utc(index)
    return index.tz_convert("UTC").tz_localize(None) + STANDARD_TIME_OFFSET


def _decimal_year(std: pd.DatetimeIndex) -> np.ndarray:
    days_in_year = np.where(std.is_leap_year, 366.0, 365.0)
    elapsed_days = (std.dayofyear - 1) + std.hour / 24.0
    return std.year + elapsed_days / days_in_year


def _design_matrix(std: pd.DatetimeIndex, n_harmonics: int) -> np.ndarray:
    decimal_year = _decimal_year(std)
    angle = 2.0 * np.pi * (decimal_year - np.floor(decimal_year))
    columns = [np.ones(len(std)), decimal_year - TREND_REFERENCE_YEAR]
    for k in range(1, n_harmonics + 1):
        columns += [np.cos(k * angle), np.sin(k * angle)]
    return np.column_stack(columns)


@dataclass(frozen=True)
class TemperatureNormal:
    """Normal temperature per standard-time hour of day: level, trend, annual harmonics.

    `coefficients[hour]` = [level, trend per year from 2000, cos1, sin1, cos2, sin2, ...].
    """

    coefficients: np.ndarray  # shape (24, 2 + 2 * n_harmonics)
    n_harmonics: int


def fit_temperature_normal(temperature_c: pd.Series, n_harmonics: int = 3) -> TemperatureNormal:
    """Least-squares fit of the normal temperature, separately for each hour of day.

    NaN hours are dropped before fitting; infinite values raise. Each hour of day
    needs at least two years of observations so level, trend and harmonics are
    identifiable.
    """
    require_utc(temperature_c.index)
    values = temperature_c.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("temperatures contain infinite values")
    observed = ~np.isnan(values)
    std = standard_time(temperature_c.index)[observed]
    values = values[observed]
    design = _design_matrix(std, n_harmonics)

    coefficients = np.empty((24, design.shape[1]))
    for hour in range(24):
        rows = np.asarray(std.hour == hour)
        if len(np.unique(std.year[rows])) < 2 or rows.sum() < 2 * design.shape[1]:
            raise ValueError(f"not enough observations to fit the normal at hour {hour}")
        coefficients[hour], *_ = np.linalg.lstsq(design[rows], values[rows], rcond=None)
    return TemperatureNormal(coefficients=coefficients, n_harmonics=n_harmonics)


def temperature_normal(model: TemperatureNormal, index: pd.DatetimeIndex) -> np.ndarray:
    """Normal temperature in °C for each hour of a UTC index."""
    std = standard_time(index)
    design = _design_matrix(std, model.n_harmonics)
    return np.einsum("ij,ij->i", design, model.coefficients[np.asarray(std.hour)])


def temperature_anomaly(temperature_c: pd.Series, model: TemperatureNormal) -> pd.Series:
    """Observed minus normal temperature in °C; NaN where the observation is NaN."""
    normal = temperature_normal(model, temperature_c.index)
    return pd.Series(
        temperature_c.to_numpy(dtype=float) - normal,
        index=temperature_c.index,
        name="temperature_anomaly_c",
    )


def german_temperature_anomaly(
    station_temperatures: Mapping[str, pd.Series],
    normals: Mapping[str, TemperatureNormal],
    weights: Mapping[str, float],
    min_reporting_weight: float,
) -> pd.Series:
    """Weighted average of per-station anomalies (each station against its own normal).

    Missing stations are handled by `weighted_temperature`: the reporting stations'
    weights are rescaled if they carry at least `min_reporting_weight`.
    """
    if set(station_temperatures) != set(normals):
        raise ValueError("stations and normals must have the same station ids")
    anomalies = {
        station_id: temperature_anomaly(series, normals[station_id])
        for station_id, series in station_temperatures.items()
    }
    return weighted_temperature(anomalies, weights, min_reporting_weight).rename(
        "temperature_anomaly_c"
    )


def weather_hours(index: pd.DatetimeIndex, weather_start_year: int) -> pd.DatetimeIndex:
    """The weather-year UTC hour used for each delivery hour.

    Matching is on month, day and hour in standard time (UTC+1). The first delivery
    year maps to `weather_start_year` and later years to the following weather years.
    A delivery 29 February uses 28 February when the weather year has none.
    """
    std = standard_time(index)
    if len(std) and ((std.minute != 0) | (std.second != 0)).any():
        raise ValueError("index must contain whole UTC hours")
    if len(std) == 0:
        return pd.DatetimeIndex([], tz="UTC")
    years = np.asarray(std.year) - std.year.min() + weather_start_year
    months, days = np.asarray(std.month), np.asarray(std.day)
    no_leap_day = ~((years % 4 == 0) & ((years % 100 != 0) | (years % 400 == 0)))
    days = np.where((months == 2) & (days == 29) & no_leap_day, 28, days)
    std_weather = pd.to_datetime(
        {"year": years, "month": months, "day": days, "hour": np.asarray(std.hour)}
    )
    utc = pd.DatetimeIndex(std_weather - STANDARD_TIME_OFFSET).tz_localize("UTC")
    return utc.rename("weather_utc")


def map_weather_year(
    series: pd.Series, index: pd.DatetimeIndex, weather_start_year: int
) -> np.ndarray:
    """Values of an hourly UTC weather series for each delivery hour; NaN where missing."""
    require_utc(series.index)
    return series.reindex(weather_hours(index, weather_start_year)).to_numpy(dtype=float)
