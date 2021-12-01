from iqradre_reader.detector.predictor import BoxesPredictor, BoxesPredictorOnnx
# from iqradre_reader.detector.predictor.onnx_predictor import BoxesPredictor
from iqradre_reader.predictor import ReaderPredictor
from pathlib import Path
from time import time
from pathlib import Path

from starlette.config import Config
config = Config(".env")

MODEL_PATH = config("MODEL_PATH", default="./ml/model/")
DETECTION_MODEL_NAME = config("DETECTION_MODEL_NAME", default="craft_idcard.onnx")
RECOGNITION_MODEL_NAME = config("RECOGNITION_MODEL_NAME", default="crnn_idcard.pth")

DETECTION_PATH = Path(MODEL_PATH).joinpath(DETECTION_MODEL_NAME)
RECOGNITION_PATH = Path(MODEL_PATH).joinpath(RECOGNITION_MODEL_NAME)


def run_test():
    image_path = '../assets/images/segment.jpg'
    config = {
        'detector': str(DETECTION_PATH),
        'recognitor': str(RECOGNITION_PATH),
    }
    
    reader = ReaderPredictor(config=config)
    
    start = time()
    result = reader.predict(image_path, auto_deskew=True)
    elapsed = time() - start
    print(result['prediction'])
    print(f'elapsed predict time: {elapsed:.4f} second' )

if __name__ == '__main__':

    run_test()
    
