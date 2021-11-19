import logging
import sys
import os
from typing import List

from core.logging import InterceptHandler
from loguru import logger
from starlette.config import Config
from starlette.datastructures import CommaSeparatedStrings, Secret



config = Config(".env")

API_PREFIX = "/api"
VERSION = "0.1.0"
DEBUG: bool = config("DEBUG", cast=bool, default=False)
MAX_CONNECTIONS_COUNT: int = config("MAX_CONNECTIONS_COUNT", cast=int, default=10)
MIN_CONNECTIONS_COUNT: int = config("MIN_CONNECTIONS_COUNT", cast=int, default=10)
SECRET_KEY: Secret = config("SECRET_KEY", cast=Secret, default="")

PROJECT_NAME: str = config("PROJECT_NAME", default="reader_service")

# logging configuration
LOGGING_LEVEL = logging.DEBUG if DEBUG else logging.INFO
logging.basicConfig(
    handlers=[InterceptHandler(level=LOGGING_LEVEL)], level=LOGGING_LEVEL
)
logger.configure(handlers=[{"sink": sys.stderr, "level": LOGGING_LEVEL}])

MODEL_PATH = config("MODEL_PATH", default="./ml/model/")
DETECTION_MODEL_NAME = config("DETECTION_MODEL_NAME", default="craft_idcard.onnx")
RECOGNITION_MODEL_NAME = config("RECOGNITION_MODEL_NAME", default="crnn_idcard.pth")
SEGMENT_MODEL_NAME = config("SEGMENT_MODEL_NAME", default="unet_idcard.onnx")

# from iqradre_reader.predictor.predictor import ReaderPredictor
# from services import loader
# reader_model = loader.load_reader_model()
