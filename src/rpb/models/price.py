"""Hourly day-ahead price model (docs/design.md section 5).

S[s, h] = level_month(h)[s] + shape(hour-of-day, day type, month) + residual[s, h]

This module holds the deterministic shape. Everything is additive in EUR/MWh:
prices can be negative, so there are no logs and no clipping.
"""

import numpy as np
import pandas as pd

from rpb.holidays import DAY_TYPES, LOCAL_TZ, day_type
from rpb.timeutils import require_utc

SHAPE_LEVELS = ("month", "day_type", "hour")


def calendar_keys(index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Local Europe/Berlin calendar keys of each hour in a UTC index."""
    local = index.tz_convert(LOCAL_TZ)
    return {
        "year": np.asarray(local.year),
        "month": np.asarray(local.month),
        "day_type": day_type(index),
        "hour": np.asarray(local.hour),  # local clock hour; hour 2 repeats on the autumn DST day
    }


def estimate_shape(price_eur_mwh: pd.Series) -> pd.Series:
    """Estimate the hourly shape from an hourly UTC price series.

    Each price is taken relative to the mean price of its local calendar month
    (per year), and the deviations are averaged by (month, day type, local hour).
    Hours with a NaN price are dropped before estimating; infinite prices raise.
    Cells with no data are NaN.

    Returns a Series `shape_eur_mwh` indexed by (month, day_type, hour), with all
    12 x 3 x 24 cells.
    """
    require_utc(price_eur_mwh.index)
    if np.isinf(price_eur_mwh.to_numpy(dtype=float)).any():
        raise ValueError("prices contain infinite values")
    prices = price_eur_mwh.dropna()
    if prices.empty:
        raise ValueError("no non-NaN prices to estimate the shape from")
    keys = calendar_keys(prices.index)

    level = prices.groupby([keys["year"], keys["month"]]).transform("mean")
    deviation = (prices - level).to_numpy()
    shape = pd.Series(deviation).groupby([keys["month"], keys["day_type"], keys["hour"]]).mean()
    cells = pd.MultiIndex.from_product([range(1, 13), DAY_TYPES, range(24)], names=SHAPE_LEVELS)
    shape.index.names = SHAPE_LEVELS
    return shape.reindex(cells).rename("shape_eur_mwh")


def evaluate_shape(shape: pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
    """Shape values in EUR/MWh for each hour of a UTC index.

    Values are re-centred to zero mean within each local calendar month present
    in `index`, so the shape never shifts the monthly level. Pass whole delivery
    months; a partial month is centred on the hours it has.
    Raises if a needed (month, day type, hour) cell is missing, NaN or infinite.
    """
    require_utc(index)
    keys = calendar_keys(index)
    cells = pd.MultiIndex.from_arrays(
        [keys["month"], keys["day_type"], keys["hour"]], names=SHAPE_LEVELS
    )
    values = shape.reindex(cells).to_numpy(dtype=float)
    missing = ~np.isfinite(values)
    if missing.any():
        examples = sorted(set(cells[missing]))[:3]
        raise ValueError(
            f"shape has no finite value for {missing.sum()} hours, e.g. cells {examples}"
        )
    month_mean = pd.Series(values).groupby([keys["year"], keys["month"]]).transform("mean")
    return values - month_mean.to_numpy()
