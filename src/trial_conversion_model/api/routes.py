import logging

from fastapi import APIRouter

from trial_conversion_model.api.schemas import PredictionRequest, PredictionResponse
from trial_conversion_model.predict import load_model

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
        # Convert request to a one-row DataFrame
    features = pd.DataFrame([request.model_dump()])

    # Predict probability of conversion
    probability = round(float(model.predict_proba(features)[0][1]), 4)

    # Assign band and return response
    band = probability_to_band(probability)

    return PredictionResponse(
        probability=probability,
        band=band,
    )
