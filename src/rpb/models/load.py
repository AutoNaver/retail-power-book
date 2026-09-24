"""Residential load model (docs/design.md section 4).

L[s, h] = N_customers · E_annual[s] · shape_h · (1 + ε[s, h])

This module holds the deterministic part (ε = 0). The shape is the BDEW H25
household profile from the `demandlib` package (MIT). BDEW publishes the profiles
without a licence, so this repository depends on `demandlib` instead of
committing copies of the data.
"""

import numpy as np
import pandas as pd
from demandlib import bdew

from rpb.holidays import LOCAL_TZ, german_public_holidays
from rpb.timeutils import require_utc


def _h25_hourly_energy(year: int) -> pd.Series:
    """H25 energy per UTC hour for one local calendar year, in arbitrary units."""
    quarter_hours = pd.date_range(
        pd.Timestamp(year, 1, 1).tz_localize(LOCAL_TZ),
        pd.Timestamp(year + 1, 1, 1).tz_localize(LOCAL_TZ),
        freq="15min",
        inclusive="left",
    )
    power_kw = bdew.H25(quarter_hours, holidays=sorted(german_public_holidays(year)))
    utc = power_kw.index.tz_convert("UTC")
    # Four equal quarter-hours: their sum is proportional to the hour's energy.
    return pd.Series(power_kw.to_numpy(), index=utc).groupby(utc.floor("h")).sum()


def profile_shape(index: pd.DatetimeIndex) -> np.ndarray:
    """H25 shape for each hour of a UTC index, summing to 1 over each local calendar year.

    A sub-range of a year sums to its share of that year's consumption. Holidays are
    the nationwide ones from `rpb.holidays`, which the profile treats as Sundays.
    """
    require_utc(index)
    if len(index) == 0:
        return np.array([], dtype=float)
    local_years = np.unique(index.tz_convert(LOCAL_TZ).year)
    shape = pd.concat([(energy := _h25_hourly_energy(int(y))) / energy.sum() for y in local_years])
    values = shape.reindex(index.tz_convert("UTC")).to_numpy()
    if np.isnan(values).any():
        raise ValueError("index has timestamps that are not whole UTC hours")
    return values


def deterministic_load_mwh(
    index: pd.DatetimeIndex, n_customers: int, annual_mwh_per_customer: float
) -> np.ndarray:
    """Load in MWh per hour under normal weather: N_customers · E_annual · shape_h."""
    if n_customers < 0 or annual_mwh_per_customer < 0:
        raise ValueError("n_customers and annual_mwh_per_customer must be non-negative")
    return n_customers * annual_mwh_per_customer * profile_shape(index)
