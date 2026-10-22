import numpy as np
import pandas as pd

from trial_conversion_model.features import DAY_COLUMNS, add_features


def make_row(
    sessions_3d: int = 3,
    listen_sessions_3d: int | None = None,
    total_minutes_3d: float | None = None,
    country: str = "US",
    device_type: str = "iOS",
) -> pd.DataFrame:
    """One raw trial snapshot, as a single-row frame ready for add_features.

    add_features re-derives sessions_3d from the per-day columns, so the total
    is split across the three days here. Listen sessions and minutes default to
    a plausible engagement level for that many sessions and can be overridden;
    a zero-session row therefore comes out completely zero.
    """
    per_day, spare = divmod(sessions_3d, len(DAY_COLUMNS))
    row = {
        "sessions_day1": per_day + int(spare > 0),
        "sessions_day2": per_day + int(spare > 1),
        "sessions_day3": per_day,
        "listen_sessions_3d": sessions_3d
        if listen_sessions_3d is None
        else listen_sessions_3d,
        "total_minutes_3d": 30.0 * sessions_3d
        if total_minutes_3d is None
        else total_minutes_3d,
        "country": country,
        "device_type": device_type,
    }
    return pd.DataFrame([row])


def test_zero_session_trial_gets_zero_shares_not_nan():
    out = add_features(make_row(sessions_3d=0))
    for col in ["day1_share", "listen_share", "avg_session_minutes"]:
        assert not out[col].isna().any(), f"{col} is NaN"
        assert (out[col] == 0).all(), f"{col} should be 0"
        assert not np.isinf(out[col]).any(), f"{col} is infinite"
