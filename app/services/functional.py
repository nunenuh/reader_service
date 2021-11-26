from services.predict import ReaderModelHandler as reader_model
from services.predict import SegmentModelHandler as segment_model
import joblib
import numpy as np
from time import time
from PIL import Image
import uuid
import os
from pathlib import Path


def predict(image, use_segment=False, auto_deskew=False, auto_resize=False):
    if use_segment:
        stime = time()
        result = segment_model.predict(image, auto_resize=auto_resize)
        image_segment = result['prediction']
        image = np.array(image_segment)
        segtime = time() - stime
    reader_predict = reader_model.predict(image, auto_deskew=auto_deskew)
    
    if use_segment:
        reader_predict['times']['segment'] = f'{segtime:.4f} s'
        
    return reader_predict


def save_images(request, images, relative_url=False):
    base_uuid = f'{uuid.uuid1().hex[:10]}'
    base_dir = f'upload_files/{base_uuid}'
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    
    urls_link = {}
    for key in images.keys():
        img = images[key]
        
        fname = f'{base_dir}/{base_uuid}_{key}.png'
        # print(f'{type(img)} Type Filename : {fname}')
        # print(f'image dtype : {img.dtype}')
        
        if type(img)==Image.Image:
            img.convert("RGB")
            img.save(fname)
            
        elif type(img)==np.ndarray:
            # print(img)
            img = Image.fromarray(img)
            img.convert("RGB")
            img.save(fname)
        
        link = f'{request.base_url}{fname}'
        if relative_url:
            link = f'{fname}'
        urls_link[key] = link
        
    return base_uuid, urls_link
    