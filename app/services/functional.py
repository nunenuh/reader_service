from services.predict import ReaderModelHandler as reader_model
from services.predict import SegmentModelHandler as segment_model
import joblib


def predict(image, use_segment=False):
    if use_segment:
        result = segment_model.predict(image)
        image = result['result']
    return reader_model.predict(image)