from pydantic import BaseModel


class PredictionRequest(BaseModel):
    """One trial's first-3-day base aggregates, as the caller knows them."""

    sessions_day1: int
    sessions_day2: int
    sessions_day3: int
    listen_sessions_3d: int
    total_minutes_3d: float
    country: str
    device_type: str


class PredictionResponse(BaseModel):
    """What we send back: a probability and a band a human can act on."""

    conversion_probability: float
    conversion_band: str
