from iqradre_reader.detector.predictor import BoxesPredictor, BoxesPredictorOnnx
# from iqradre_reader.detector.predictor.onnx_predictor import BoxesPredictor
from iqradre_reader.predictor import ReaderPredictor
from pathlib import Path
from time import time

if __name__ == '__main__':

    image_path = 'assets/images/segment.jpg'
    config = {
        'detector': 'ml/model/craft_idcard.onnx',
        'recognitor': 'ml/model/crnn_idcard.pth',
    }
    
    reader = ReaderPredictor(config=config)
    
    start = time()
    result = reader.predict(image_path, auto_deskew=True)
    elapsed = time() - start
    print(result['prediction'])
    print(f'elapsed predict time: {elapsed:.4f} second' )
    
