import os
import sys
import geopandas as gp
import pandas as pd
import argparse
from shapely.strtree import STRtree
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
from scipy.optimize import linear_sum_assignment
from keras_retinanet import models

sys.path.insert(0, os.path.abspath('..'))

from DeepForest import utils, preprocess
from DeepForest.config import load_config
from DeepForest import h5_generator
from DeepForest import evalmAP
from DeepForest.utils.generators import create_NEON_generator

parser     = argparse.ArgumentParser(description='Prediction of a new image')
parser.add_argument('--model', help='path to training model' )
parser.add_argument('--batch_size', help='batch size for prediction', default=1)

args=parser.parse_args()

#Config files
DeepForest_config = load_config(dir="..")

#Load hand annotations
neon_generator = create_NEON_generator(args.batch_size, DeepForest_config)

#Get detections and annotations
model = models.load_model(args.model, backbone_name='resnet50', convert=True, nms_threshold=DeepForest_config["nms_threshold"])
labels_to_names = {0: 'Tree'}
all_detections = evalmAP._get_detections(neon_generator, model, score_threshold=DeepForest_config["score_threshold"], save_path ="/Users/Ben/Downloads/")
all_annotations = evalmAP._get_annotations(neon_generator)

#Loop through images and match boxes.
area_results = []

for i in range(neon_generator.size()):
    detections           = all_detections[i][0]
    annotations          = all_annotations[i][0]
    
    #Turn to polygons
    prediction_polygons = utils.boxes_to_polygon(detections)
    annotation_polygons = utils.boxes_to_polygon(annotations)
    
    #Create overlap matrix
    cost_matrix=np.zeros((len(annotation_polygons), len(prediction_polygons)))

    for x, poly in enumerate(annotation_polygons):    
        for y, match in enumerate(prediction_polygons):
            cost_matrix[x,y]= poly.intersection(match).area

    #Assign polygon pairs
    assignments = linear_sum_assignment(-1 *cost_matrix)
    
    for k in np.arange(len(assignments[0])):        
        annotation=annotation_polygons[assignments[0][k]]
        overlapping_prediction=prediction_polygons[assignments[1][k]]
        
        #If the assignment doesn't overlap by more than 10%, set to 0 for bad prediction
        if overlapping_prediction.intersection(annotation).area/overlapping_prediction.area < 0.10:
            area_results.append([0, annotation.area])         
        else:
            #plot if ugly
            #if (overlapping_prediction.area - annotation.area )/ overlapping_prediction.area * 100 > 50:
                
                #image = neon_generator.load_image(i)
                #fig, ax = plt.subplots(1)
                
                ##prediction
                #x, y = overlapping_prediction.exterior.coords.xy
                #points = np.array([x, y], np.int32).T
                #polygon_shape = patches.Polygon(points, linewidth=1, edgecolor='r', facecolor='none')
                #ax.add_patch(polygon_shape)
                
                ##annotation
                #x, y = annotation.exterior.coords.xy
                #points = np.array([x, y], np.int32).T
                #polygon_shape = patches.Polygon(points, linewidth=1, edgecolor='blue', facecolor='none')
                #ax.add_patch(polygon_shape)
                
                #ax.imshow(image[:,:,::-1])   
                #plt.show(block=True)
                
            #get area 
            area_results.append([overlapping_prediction.area, annotation.area])

area_df =pd.DataFrame(area_results, columns=["prediction","annotation"])    
site = DeepForest_config["evaluation_site"][0]

filname = "prediction_area" + "_" + site + "_"  + args.model + ".csv"
area_df.to_csv(filname)

