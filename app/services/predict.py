import os

from core.errors import PredictException, ModelLoadException
from core.config import MODEL_PATH, RECOGNITION_MODEL_NAME, DETECTION_MODEL_NAME, SEGMENT_MODEL_NAME

from core.config import TEXT_THRESHOLD, LINK_THRESHOLD, LOW_TEXT, MIN_SIZE_PERCENT
from loguru import logger

from iqradre_reader.predictor.predictor import ReaderPredictor
from iqradre_segment.prod import SegmentationPredictorOnnx

import joblib

class ReaderModelHandler(object):
    model = None

    @classmethod
    def predict(cls, input, auto_deskew=False, load_wrapper=joblib.load, method="predict"):
        clf = cls.get_model(load_wrapper)
        if hasattr(clf, method):
            print(f"ReaderHandler: predict params:")
            print(f"params: TEXT_THRESHOLD: {TEXT_THRESHOLD}")
            print(f"params: LINK_THRESHOLD: {LINK_THRESHOLD}")
            print(f"params: LOW_TEXT: {LOW_TEXT}")
            print(f"params: MIN_SIZE_PERCENT: {MIN_SIZE_PERCENT}")
            
            return getattr(clf, method)(
                input,
                text_threshold=TEXT_THRESHOLD,
                link_threshold=LINK_THRESHOLD,
                low_text=LOW_TEXT,
                min_size_percent=MIN_SIZE_PERCENT,
                auto_deskew=auto_deskew
            )
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

class SegmentModelHandler(object):
    model = None

    @classmethod
    def predict(cls, input, auto_resize=True, load_wrapper=joblib.load, method="predict"):
        clf = cls.get_model(load_wrapper)
        if hasattr(clf, method):
            return getattr(clf, method)(input, auto_resize=auto_resize)
        raise PredictException(f"'{method}' attribute is missing")

    @classmethod
    def get_model(cls, load_wrapper):
        if cls.model is None and load_wrapper:
            cls.model = cls.load(load_wrapper)
        return cls.model

    @staticmethod
    def load(load_wrapper) -> SegmentationPredictorOnnx:
        model = None
        if MODEL_PATH.endswith("/"):
            path = f"{MODEL_PATH}{SEGMENT_MODEL_NAME}"
        else:
            path = f"{MODEL_PATH}/{SEGMENT_MODEL_NAME}"
            
        if not (os.path.exists(path)):
            message = f"Machine learning model at {path} not exists!"
            logger.error(message)
            raise FileNotFoundError(message)
        
        # model = load_wrapper(path)

        model = SegmentationPredictorOnnx(weight_path=path)
        
        if not model:
            message = f"Model {model} could not load!"
            logger.error(message)
            raise ModelLoadException(message)

        return model