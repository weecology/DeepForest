# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

#parse args
import argparse

#DeepForest
from DeepForest import onthefly_generator
from DeepForest.preprocess import compute_windows, retrieve_window
import pandas as pd
from PIL import Image


#Set training or training
mode_parser     = argparse.ArgumentParser(description='Prediction of a new image')
mode_parser.add_argument('--model', help='path to training model' )
mode_parser.add_argument('--image', help='image to predict' )
mode_parser.add_argument('--single_image', help='predict image in a single batch',action="store_true")

args=mode_parser.parse_args()

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases

# load retinanet model
# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
model = models.load_model(args.model, backbone_name='resnet50', convert=True,nms_threshold=0.1)

labels_to_names = {0: 'Tree'}

if args.single_image:
    # load image
    image = read_image_bgr(args.image)
    
    # copy to draw on
    draw = image.copy()
    #draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    
    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)
    
    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)
    
    # correct for image scale
    boxes /= scale
    
    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.1:
            break
            
        color = label_color(label)
        
        b = box.astype(int)
        draw_box(draw, b, color=color)
        
        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)
        
    cv2.imwrite("/Users/Ben/Downloads/test.png",draw)
else:
    
    from DeepForest.config import load_config
    DeepForest_config=load_config("train")
    
    windows=compute_windows(image=args.image, pixels=DeepForest_config["patch_size"], overlap=DeepForest_config["patch_overlap"])
    
    #Compute Windows
    tile_windows={}
    
    tile_windows["image"]=os.path.split(args.image)[-1]
    tile_windows["windows"]=np.arange(0,len(windows))
    
    numpy_image=read_image_bgr(args.image)
        
    for index in tile_windows["windows"]:
        
        image=retrieve_window(numpy_image, index, windows)
        
        # copy to draw on
        draw = image.copy()
        
        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image)

        # process image
        start = time.time()
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        print("processing time: ", time.time() - start)
        
        # correct for image scale
        boxes /= scale
        
        # visualize detections
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score < 0.2:
                break
                
            color = label_color(label)
            
            b = box.astype(int)
            draw_box(draw, b, color=color)
            
            caption = "{} {:.3f}".format(labels_to_names[label], score)
            draw_caption(draw, b, caption)
            cv2.imwrite("/Users/Ben/Downloads/" + os.path.splitext(tile_windows["image"])[0]+ "_" +  str(index) + ".png",draw)
    
    
#plt.figure(figsize=(15, 15))
#plt.axis('off')
#plt.imshow(draw)
#plt.show()
