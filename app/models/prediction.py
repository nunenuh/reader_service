from pydantic import BaseModel
from typing import Optional


class MachineLearningResponse(BaseModel):
    prediction: list
    times: dict
    urls: Optional[dict]
    index: Optional[str]


class HealthResponse(BaseModel):
    status: bool
