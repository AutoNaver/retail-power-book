"""Checks for the timezone contract: timestamps are stored as timezone-aware UTC."""

import pandas as pd


def is_utc(index: pd.DatetimeIndex) -> bool:
    """True if `index` is timezone-aware with a zero UTC offset at every timestamp.

    This accepts any UTC alias ("UTC", "Etc/UTC", "GMT", `datetime.timezone.utc`)
    by checking offsets rather than the zone's name.
    """
    if index.tz is None:
        return False
    local_wall = index.tz_localize(None).asi8
    utc_wall = index.tz_convert("UTC").tz_localize(None).asi8
    return bool((local_wall == utc_wall).all())


def require_utc(index: pd.DatetimeIndex) -> None:
    """Raise ValueError unless `index` is UTC (see `is_utc`) and has no missing timestamps."""
    if index.hasnans:
        raise ValueError("index contains NaT; missing timestamps would be silently misclassified")
    if not is_utc(index):
        raise ValueError("index must be timezone-aware UTC")
