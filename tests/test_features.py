import pandas as pd
import numpy as np

from trial_conversion_model.features import add_features


def test_zero_session_trial_gets_zero_shares_not_nan():
    out = add_features(make_row(sessions_3d=0))
    for col in ["day1_share", "listen_share", "avg_session_minutes"]:
        assert not out[col].isna().any(), f"{col} is NaN"
        assert (out[col] == 0).all(), f"{col} should be 0"
        assert not np.isinf(out[col]).any(), f"{col} is infinite"
