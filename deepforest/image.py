"""
Utility functions for reading/cleaning images and generators
"""
import cv2
import keras
import numpy as np

def equalize(img):
    """Equalize color array for RGB image"""
    img = img.astype(np.uint8)
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8))
    
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels
    
    l2 = clahe.apply(l)  # apply CLAHE to the L-channel
    
    lab = cv2.merge((l2,a,b))  # merge channels
    img_equalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return img_equalized
    
def normalize(image):
    """Mean normalization across dataset"""
    
    #RGB - Imagenet means
    image = image.astype(keras.backend.floatx())    
    image[..., 0] -= 103.939
    image[..., 1] -= 116.779
    image[..., 2] -= 123.68
    
    return image

def preprocess(image):
    """Preprocess an image for retinanet"""
    
    #equalize histogram
    image = equalize(image)
    
    #mean normalize
    image = normalize(image)
    
    return image
    
