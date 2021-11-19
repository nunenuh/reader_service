from typing import Any

import joblib
from core.errors import PredictException
from fastapi import APIRouter, HTTPException
from loguru import logger
from models.prediction import HealthResponse, MachineLearningResponse
from services import functional as predictor
from fastapi import FastAPI, File, UploadFile, Request
from libml import utils


router = APIRouter()

# get_prediction = lambda image: model.predict(image, load_wrapper=joblib.load, method="predict")


@router.post("/predict", response_model=MachineLearningResponse, name="predict:text-reader")
async def predict(file: UploadFile = File(...), use_segment=False):
    if not file:
        raise HTTPException(status_code=404, detail=f"'data_input' argument invalid!")
    
    if not (file.content_type == "image/jpeg" or file.content_type=="image/png"):
        raise HTTPException(status_code=404, detail=f"'content_type' is not accepted, send only jpg or png image!")
         
    try:
        file_contents = await file.read()
        image = utils.convert_buffer_to_npimage(file_contents)
        result = predictor.predict(image, use_segment=use_segment)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Exception: {e}")

    return MachineLearningResponse(prediction=result['prediction'], times=result['times'])
    
