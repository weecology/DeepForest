#utility functions for demo
import os
import yaml
import sys
import keras
import cv2
import numpy as np
import copy
import pandas as pd
import glob
from tqdm import tqdm
import json
import urllib

#DeepForest
from keras_retinanet import models
from keras_retinanet.utils import image as keras_retinanet_image
from keras_retinanet.utils.visualization import draw_detections

def label_to_name(label):
        """ Map label to name.
        """
        return "Tree"

def read_config():
        try:
                with open("deepforest_config.yml", 'r') as f:
                        config = yaml.load(f)
        except Exception as e:
                raise FileNotFoundError("There is no config file in dir:{}, yields {}".format(os.getcwd(),e))
                
        return config

def read_model(model_path, config):
        model = models.load_model(model_path, backbone_name='resnet50')
        return model

#Download progress bar
class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                        self.total = tsize
                self.update(b * bsize - self.n)
                
def download_release():
        """Download the latest tag model from github and save it the /data folder
        Returns
        -------
         str
                 Path to model weights
        """
        #Find latest github tag release from the DeepLidar repo
        _json = json.loads(urllib.request.urlopen(urllib.request.Request(
                'https://api.github.com/repos/Weecology/DeepForest/releases/latest',
            headers={'Accept': 'application/vnd.github.v3+json'},
             )).read())        
        asset = _json['assets'][0]
        output_path = os.path.join('data',asset['name'])    
        url = asset['browser_download_url']
        
        #Download if it doesn't exist
        if not os.path.exists(output_path):
                with DownloadProgressBar(unit='B', unit_scale=True,
                                     miniters=1, desc=url.split('/')[-1]) as t:
                        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)           
        
        return output_path

def predict_image(model, image_path, score_threshold = 0.1, max_detections= 200, return_plot=True):
        """
        Predict an image
        return_plot: Logical. If true, return a image object, else return bounding boxes
        """
        #predict
        raw_image = cv2.imread(image_path)        
        image        = image_utils.preprocess_image(raw_image)
        image, scale = keras_retinanet_image.resize_image(image)

        if keras.backend.image_data_format() == 'channels_first':
                image = image.transpose((2, 0, 1))

        # run network
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))[:3]

        # correct boxes for image scale
        boxes /= scale

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]

        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        image_boxes      = boxes[0, indices[scores_sort], :]
        image_scores     = scores[scores_sort]
        image_labels     = labels[0, indices[scores_sort]]
        image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)
        
        if return_plot:
                draw_detections(raw_image, image_boxes, image_scores, image_labels, label_to_name=label_to_name, score_threshold=score_threshold)
                return raw_image                
        else:
                return image_boxes

def prediction_wrapper(image_path):
        ##read model
        config = read_config()        
        model = read_model(config["model_path"], config) 
        prediction = predict_image(model, image_path, score_threshold = 0.1, max_detections= 200,return_plot=True)
        save_name = os.path.splitext(image_path)[0] + "_prediction.jpg"
        cv2.imwrite(save_name,prediction)
        return(save_name)
                
def predict_all_images():
        """
        loop through a dir and run all images
        """
        #Read config
        config = read_config()
        
        #read model
        model = read_model(config["model_path"], config)
        tifs = glob.glob(os.path.join("data","**","*.tif"))
        for tif in tifs:
                print(tif)
                prediction = predict_image(model, tif, score_threshold = 0.1, max_detections= 200,return_plot=False)
                
                #reshape and save to csv
                df = pd.DataFrame(prediction)
                df.columns = ["xmin","ymin","xmax","ymax"]
                
                #save boxes
                file_path = os.path.splitext(tif)[0] + ".csv"
                df.to_csv(file_path)
