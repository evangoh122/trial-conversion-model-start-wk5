import logging

import pandas as pd
from fastapi import APIRouter

from trial_conversion_model.api.schemas import PredictionRequest, PredictionResponse
from trial_conversion_model.predict import load_model, predict_proba

logger = logging.getLogger(__name__)

router = APIRouter()

# Load the model once, when the service starts, not on every request.
model = load_model()


def to_band(probability: float) -> str:
    """Turn a raw probability into a label a human can act on."""
    if probability < 0.33:
        return "low"
    if probability < 0.66:
        return "medium"
    return "high"


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict conversion probability for a single live trial."""
    # The caller sends base aggregates; predict_proba adds the derived
    # features and lines the dummy columns up with the model's training set.
    features = pd.DataFrame([request.model_dump()])

    probability = round(float(predict_proba(model, features).iloc[0]), 4)
    band = to_band(probability)

    return PredictionResponse(
        conversion_probability=probability,
        conversion_band=band,
    )
