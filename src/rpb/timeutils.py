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
