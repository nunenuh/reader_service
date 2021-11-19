import PIL
from PIL import Image
import numpy as np
import cv2 as cv

def convert_buffer_to_npimage(contents):
    nparr = np.fromstring(contents, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img

def convert_npimage_to_pilimage(image: np.ndarray):
    out_image = Image.fromarray(image)
    out_image = PIL.ImageOps.exif_transpose(out_image)
    return out_image