import os
import cv2
import numpy as np
from torch import Tensor

def mask2bounding_rect(mask):
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 1:
        return 0, 0, 0, 0
    cont = contours[0]
    x, y, w, h = cv2.boundingRect(cont)
    return x, y, w, h

def mask_and_crop_ori_image(image, mask):
    
    overlay_image = (image * np.where(mask>0, 1, 0)).astype(np.uint8)
    x, y, w, h = mask2bounding_rect(mask)
    square_a = min(w, h)
    x = x+w//2 - square_a // 2
    y = y+h//2 - square_a // 2
    croped_image = overlay_image[y:y+square_a, x:x+square_a]
    croped_image = cv2.resize(croped_image.astype(np.uint8), (512,512))
    return croped_image

def min_max_normalization(image):
    assert isinstance(image, Tensor), 'image should be a tensor for min-max normalization'
    return (image - image.min()) / (image.max() - image.min())