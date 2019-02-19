import argparse
from DeepForest import utils, preprocess
from DeepForest.config import load_config
from DeepForest import h5_generator
from train import create_NEON_generator
from DeepForest import evalmAP
from keras_retinanet import models
import geopandas as gp
import pandas as pd
from shapely.strtree import STRtree
import numpy as np

parser     = argparse.ArgumentParser(description='Prediction of a new image')
parser.add_argument('--model', help='path to training model' )
parser.add_argument('--batch_size', help='batch size for prediction', default=1)

args=parser.parse_args()

#Config files
DeepForest_config = load_config()

#Load hand annotations
neon_generator = create_NEON_generator(args, DeepForest_config["evaluation_site"], DeepForest_config)

#Get detections and annotations
model = models.load_model(args.model, backbone_name='resnet50', convert=True, nms_threshold=DeepForest_config["nms_threshold"])
labels_to_names = {0: 'Tree'}
all_detections = evalmAP._get_detections(neon_generator, model, score_threshold=DeepForest_config["score_threshold"])
all_annotations = evalmAP._get_annotations(neon_generator)

#Loop through images and match boxes.
area_results = []

for i in range(neon_generator.size()):
    detections           = all_detections[i][0]
    annotations          = all_annotations[i][0]
    
    #Turn to polygons
    prediction_polygons = utils.boxes_to_polygon(detections)
    annotation_polygons = utils.boxes_to_polygon(annotations)
    
    #Create pandas object to hold area calculation
    area_df = pd.DataFrame()
    
    #create prediction spatial index
    #from shapely.strtree import STRtree
    spatial_index = STRtree(prediction_polygons)
    
    for annotation in annotation_polygons:
        results = spatial_index.query(annotation)
        
        #for each result calculate area of overlap
        areas = []
        for result in results:
            areas.append(annotation.intersection(result).area)
            
        if len(areas) > 0:
            #select best overlap
            overlapping_prediction = results[np.argmax(areas)]
            
            #get area 
            area_results.append([overlapping_prediction.area, annotation.area])
        else:
            area_results.append([0, annotation.area])
    
area_df =pd.DataFrame(area_results, columns=["prediction","annotation"])    
area_df.to_csv("/Users/Ben/Dropbox/Weecology/NEON/prediction_area_SJER_fullmodel.csv")

