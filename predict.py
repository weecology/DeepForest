import keras
import pyfor
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

#parse args
import argparse
import glob

#DeepForest
from DeepForest import onthefly_generator, config
from DeepForest.preprocess import compute_windows, retrieve_window
from DeepForest import postprocessing, utils
import pandas as pd
from PIL import Image

#Set training or training
mode_parser     = argparse.ArgumentParser(description='Prediction of a new image')
mode_parser.add_argument('--model', help='path to training model' )
mode_parser.add_argument('--image', help='image or directory of images to predict' )
mode_parser.add_argument('--score_threshold', default=0.25)
mode_parser.add_argument('--nms_threshold', default=0.1)
mode_parser.add_argument('--output_dir', default="snapshots/images/")

args=mode_parser.parse_args()

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

#load config
DeepForest_config = config.load_config()
# adjust this to point to your downloaded/trained model

# load retinanet model
model = models.load_model(args.model, backbone_name='resnet50', convert=True, nms_threshold=args.nms_threshold)
labels_to_names = {0: 'Tree'}

if os.path.isdir(args.image):
    images=glob.glob(os.path.join(args.image,"*.tif"))
else:
    images=args.image
    
for image_path in images:
    print(image_path)

    # load image
    image = read_image_bgr(image_path)
    
    # copy to draw on
    draw = image.copy()
    
    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(img = image, min_side = 400)
    
    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)
    
    # correct for image scale
    boxes /= scale
       
    # visualize ld detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < args.score_threshold:
            break
            
        color = label_color(label)
        
        b = box.astype(int)
        draw_box(draw, b, color=(255,0,0))
        
        caption = "{} {:.2f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)
        
    #only pass score threshold boxes
    quality_boxes = []
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        quality_boxes.append(box)
        # scores are sorted so we can break
        if score < args.score_threshold:
            break
    
    #drape boxes
    #get image name
    image_name = os.path.splitext(os.path.basename(image_path))[0]     
    pc = postprocessing.drape_boxes(boxes=quality_boxes, tilename=image_name, lidar_dir=DeepForest_config["lidar_path"])
    
    #Skip if point density is too low    
    if pc:
        #Get new bounding boxes
        new_boxes = postprocessing.cloud_to_box(pc)    
        #expends 3dim
        new_boxes = np.expand_dims(new_boxes, 0)
        
        # visualize detections
        for box, score, label in zip(new_boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score < args.score_threshold:
                break
                
            color = label_color(label)
            
            b = box.astype(int)
            draw_box(draw, b, color=(0,90,255))
            
            #caption = "{} {:.2f}".format(labels_to_names[label], score)
            #draw_caption(draw, b, caption)    
    
    #Write .shp of predictions?
    filename =  os.path.join(args.output_dir, image_name + ".tif")
    cv2.imwrite(filename, draw)

#plt.figure(figsize=(15, 15))
#plt.axis('off')
#plt.imshow(draw)
#plt.show()