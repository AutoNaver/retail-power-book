"""Hourly delivery calendar and DE-LU baseload/peakload forward products.

All indexes are timezone-aware UTC. Delivery periods are defined in local
Europe/Berlin time and converted, so DST days have 23 or 25 hours.
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

import numpy as np
import pandas as pd

from rpb.timeutils import is_utc

LOCAL_TZ = "Europe/Berlin"
PEAK_START_HOUR = 8
PEAK_END_HOUR = 20


def _local_midnight_utc(day: date) -> pd.Timestamp:
    if isinstance(day, datetime):
        raise TypeError("pass a date, not a datetime: delivery periods start at local midnight")
    # Midnight is never ambiguous or missing in Europe/Berlin (DST switches at 02:00/03:00).
    return pd.Timestamp(day).tz_localize(LOCAL_TZ).tz_convert("UTC")


def delivery_index(start: date, end: date) -> pd.DatetimeIndex:
    """Hourly UTC index of delivery-hour starts for the local period [start, end)."""
    start_utc = _local_midnight_utc(start)
    end_utc = _local_midnight_utc(end)
    if end_utc <= start_utc:
        raise ValueError(f"end {end} must be after start {start}")
    return pd.date_range(start_utc, end_utc, freq="h", inclusive="left", name="delivery_start_utc")


@dataclass(frozen=True)
class Product:
    """A DE-LU forward product: base or peak, delivering over one month or one quarter."""

    kind: Literal["base", "peak"]
    year: int
    period: Literal["M", "Q"]
    number: int  # month 1-12 or quarter 1-4

    def __post_init__(self) -> None:
        if self.kind not in ("base", "peak"):
            raise ValueError(f"kind must be 'base' or 'peak', got {self.kind!r}")
        if self.period not in ("M", "Q"):
            raise ValueError(f"period must be 'M' or 'Q', got {self.period!r}")
        n_max = 12 if self.period == "M" else 4
        if not 1 <= self.number <= n_max:
            raise ValueError(f"{self.period} number must be in 1..{n_max}, got {self.number}")

    @property
    def name(self) -> str:
        number = f"{self.number:02d}" if self.period == "M" else str(self.number)
        return f"DE-{self.kind.upper()}-{self.year}-{self.period}{number}"

    @property
    def start(self) -> date:
        first_month = self.number if self.period == "M" else 3 * (self.number - 1) + 1
        return date(self.year, first_month, 1)

    @property
    def end(self) -> date:
        """First local day after delivery."""
        n_months = 1 if self.period == "M" else 3
        month_index = self.start.month - 1 + n_months
        return date(self.year + month_index // 12, month_index % 12 + 1, 1)

    def months(self) -> list["Product"]:
        """The monthly products this product delivers over (itself if monthly)."""
        if self.period == "M":
            return [self]
        return [Product(self.kind, self.year, "M", self.start.month + i) for i in range(3)]


def product_mask(product: Product, index: pd.DatetimeIndex) -> np.ndarray:
    """Boolean mask of the hours in `index` (UTC hour starts) that `product` delivers.

    Peak is Monday-Friday 08:00-20:00 Europe/Berlin. Public holidays are not
    excluded; see open question Q1 in docs/design.md before relying on this.
    """
    if not is_utc(index):
        raise ValueError("index must be timezone-aware UTC")
    in_period = (index >= _local_midnight_utc(product.start)) & (
        index < _local_midnight_utc(product.end)
    )
    if product.kind == "peak":
        local = index.tz_convert(LOCAL_TZ)
        in_period &= (
            (local.dayofweek < 5) & (local.hour >= PEAK_START_HOUR) & (local.hour < PEAK_END_HOUR)
        )
    return np.asarray(in_period, dtype=bool)
