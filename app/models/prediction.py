from pydantic import BaseModel
from typing import Optional


class MachineLearningResponse(BaseModel):
    prediction: list
    times: dict


class HealthResponse(BaseModel):
    status: bool
