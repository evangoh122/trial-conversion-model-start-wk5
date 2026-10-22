from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

from trial_conversion_model.features import CATEGORICAL, FEATURES, add_features

MODEL_PATH = Path("models/model.json")

# The cut-offs a human uses to triage a score. They live next to the
# prediction because the dashboard labels rows with the same bands the API
# returns.
BAND_LOW = 0.33
BAND_HIGH = 0.66


def to_band(probability: float) -> str:
    """Turn a raw probability into a label a human can act on."""
    if probability < BAND_LOW:
        return "low"
    if probability < BAND_HIGH:
        return "medium"
    return "high"


def load_model(path: Path = MODEL_PATH) -> XGBClassifier:
    """Load the trained model from its native XGBoost artifact."""
    model = XGBClassifier()
    model.load_model(path)
    return model


def predict_proba(model: XGBClassifier, aggregates: pd.DataFrame) -> pd.Series:
    """Score trials from their base aggregates; one probability per row.

    The rows go through the same add_features as training. A small batch
    rarely carries every country and device, so the dummy columns are
    reindexed against the model's training columns; the categories the
    batch does not have become explicit zeros.
    """
    df = add_features(aggregates)
    rows = pd.get_dummies(df[FEATURES], columns=CATEGORICAL)
    rows = rows.reindex(columns=model.get_booster().feature_names, fill_value=0)
    return pd.Series(model.predict_proba(rows)[:, 1], index=aggregates.index)
