import os

from core.errors import PredictException, ModelLoadException
from core.config import MODEL_PATH, RECOGNITION_MODEL_NAME, DETECTION_MODEL_NAME
from loguru import logger

from iqradre_reader.predictor.predictor import ReaderPredictor
import joblib



class MachineLearningModelHandler(object):
    model = None

    @classmethod
    def predict(cls, input, load_wrapper=joblib.load, method="predict"):
        clf = cls.get_model(load_wrapper)
        if hasattr(clf, method):
            return getattr(clf, method)(input)
        raise PredictException(f"'{method}' attribute is missing")

    @classmethod
    def get_model(cls, load_wrapper):
        if cls.model is None and load_wrapper:
            cls.model = cls.load(load_wrapper)
        return cls.model

    @staticmethod
    def load(load_wrapper) -> ReaderPredictor:
        model = None
        if MODEL_PATH.endswith("/"):
            detect_path = f"{MODEL_PATH}{DETECTION_MODEL_NAME}"
            recog_path = f"{MODEL_PATH}{RECOGNITION_MODEL_NAME}"
        else:
            detect_path = f"{MODEL_PATH}/{DETECTION_MODEL_NAME}"
            recog_path = f"{MODEL_PATH}/{RECOGNITION_MODEL_NAME}"
            
        if not (os.path.exists(detect_path) or os.path.exists(recog_path)):
            message = f"Machine learning model at {detect_path} or {recog_path} not exists!"
            logger.error(message)
            raise FileNotFoundError(message)
        
        # model = load_wrapper(path)
        
                
        config = {
            'detector': detect_path,
            'recognitor': recog_path,
        }

        model = ReaderPredictor(config=config)
        
        if not model:
            message = f"Model {model} could not load!"
            logger.error(message)
            raise ModelLoadException(message)

        return model
