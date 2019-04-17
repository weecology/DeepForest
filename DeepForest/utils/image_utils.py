"""
Utility functions for reading/cleaning images and generators
"""
import numpy as np
import keras
import matplotlib.pyplot as plt
import os
import numpy as np
import time
import pandas as pd
import geopandas as gp
import tensorflow as tf
import rtree
import argparse
import glob

from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from shapely import geometry

#DeepForest
from DeepForest import config
from DeepForest.preprocess import compute_windows, retrieve_window, NEON_annotations
from DeepForest import postprocessing

#Utility functions

def proportion_NA(pc):
    """Returns the proportion (%) of cells that are nan in a canopy raster array"""
    chm=pc.chm(cell_size=1)
    proportionNA = np.count_nonzero(np.isnan(chm.array))/chm.array.size
    return proportionNA * 100
    
def normalize(image):
    """Max normalization across sample"""
    #RGB
    image = image.astype(keras.backend.floatx())    
    image[..., 0] -= 103.939
    image[..., 1] -= 116.779
    image[..., 2] -= 123.68
    
    #Height model
    image[:,:,3] = image[:,:,3] 
    
    return image


def image_is_blank(image):
    
    is_zero=image.sum(2)==0
    is_zero=is_zero.sum()/is_zero.size
    
    if is_zero > 0.05:
        return True
    else:
        return False

def expand_grid(data_dict):
    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())

def box_overlap(row, window):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    window : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    box : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    
    #construct box
    box={}

    #top left
    box["x1"]=row["origin_xmin"]
    box["y1"]=row["origin_ymin"]

    #Bottom right
    box["x2"]=row["origin_xmax"]
    box["y2"]=row["origin_ymax"]     
    
    assert window['x1'] < window['x2']
    assert window['y1'] < window['y2']
    assert box['x1'] < box['x2']
    assert box['y1'] < box['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(window['x1'], box['x1'])
    y_top = max(window['y1'], box['y1'])
    x_right = min(window['x2'], box['x2'])
    y_bottom = min(window['y2'], box['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    window_area = (window['x2'] - window['x1']) * (window['y2'] - window['y1'])
    box_area = (box['x2'] - box['x1']) * (box['y2'] - box['y1'])

    overlap = intersection_area / float(box_area)
    return overlap

def _discrete_cmap(n_bin, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, n_bin))
    cmap_name = base.name + str(n_bin)
    return LinearSegmentedColormap.from_list(cmap_name, color_list, n_bin)

def boxes_to_polygon(boxes):
    """Turn the results of a retinanet bounding box array to a shapely geometry object"""
    
    polygons = []
    
    for box in boxes:
        
        #Construct polygon
        xmin = box[0]
        ymin = box[1]
        xmax = box[2]
        ymax = box[3]        
        
        top_left = [xmin, ymin]
        top_right = [xmax, ymin]
        bottom_left = [xmin, ymax]
        bottom_right = [xmax, ymax]
        polygon = geometry.Polygon([top_left, bottom_left, bottom_right, top_right])
        polygons.append(polygon)
        
    polygons = gp.GeoSeries(polygons)
    
    return polygons

def match_polygons(prediction_df, annotation_df):
    """
    Match polygons (shapely objects) based on area of overlap
    """

    '''
    1) Find overlap among polygons efficiently 
    2) Calulate a cost matrix of overlap, with rows as itcs and columns as predictions
    3) Hungarian matching for pairing
    4) Calculate intersection over union (IoU)
    '''
    
    #for each annotation, which predictions overlap
    possible_matches_index = list(prediction_df.sindex.intersection(annotation_df.geometry[0].bounds))
    
    possible_matches_index = list(prediction_df.sindex.intersection(annotation_df.bounds[0]))

    for pos, cell in enumerate(prediction_df.geometry):
        # assuming cell is a shapely object
        idx.insert(pos, cell.bounds)

    overlap_dict={}

    #select predictions that overlap with the polygons
    matched=[prediction_df.geometry[x] for x in idx.intersection(prediction_df.geometry.bounds)]

    #Create a container
    cost_matrix=np.zeros((len(itc_polygons),len(matched)))

    for x,poly in enumerate(itc_polygons):    
        for y,match in enumerate(matched):
            cost_matrix[x,y]= poly.intersection(match).area

    #Assign polygon pairs
    assignments=linear_sum_assignment(-1 *cost_matrix)

    iou_list=[]

    for i in np.arange(len(assignments[0])):        
        a=itc_polygons[assignments[0][i]]
        b=matched[assignments[1][i]]
        iou=IoU_polygon(a,b)
        iou_list.append(iou)

    return(iou_list)