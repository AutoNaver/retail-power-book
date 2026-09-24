"""German nationwide public holidays and day types for the price shape.

Only holidays observed in every federal state are included. Regional holidays
(e.g. Epiphany, Corpus Christi, All Saints' Day) differ by state and are left out,
as are 24 and 31 December, which are not public holidays.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from rpb.timeutils import require_utc

LOCAL_TZ = "Europe/Berlin"
DAY_TYPES = ("weekday", "saturday", "sunday_holiday")


def easter_sunday(year: int) -> date:
    """Gregorian Easter Sunday (anonymous Gregorian algorithm; names follow its usual notation)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def german_public_holidays(year: int) -> set[date]:
    """Nationwide German public holidays in `year`."""
    easter = easter_sunday(year)
    holidays = {
        date(year, 1, 1),  # Neujahr
        easter - timedelta(days=2),  # Karfreitag
        easter + timedelta(days=1),  # Ostermontag
        date(year, 5, 1),  # Tag der Arbeit
        easter + timedelta(days=39),  # Christi Himmelfahrt
        easter + timedelta(days=50),  # Pfingstmontag
        date(year, 10, 3),  # Tag der Deutschen Einheit
        date(year, 12, 25),  # 1. Weihnachtstag
        date(year, 12, 26),  # 2. Weihnachtstag
    }
    if year == 2017:
        holidays.add(date(2017, 10, 31))  # 500 years of the Reformation, nationwide once
    return holidays


def day_type(index: pd.DatetimeIndex) -> np.ndarray:
    """Day type of each hour in a UTC index, by its local Europe/Berlin date.

    Returns "weekday", "saturday" or "sunday_holiday"; a holiday on a Saturday
    counts as "sunday_holiday".
    """
    require_utc(index)
    local = index.tz_convert(LOCAL_TZ)
    local_dates = local.date
    holidays: set[date] = set()
    for year in np.unique(local.year):
        holidays |= german_public_holidays(int(year))
    is_holiday = np.fromiter((d in holidays for d in local_dates), dtype=bool, count=len(index))
    weekday = np.asarray(local.dayofweek)
    return np.where(
        is_holiday | (weekday == 6),
        "sunday_holiday",
        np.where(weekday == 5, "saturday", "weekday"),
    ).astype(object)
