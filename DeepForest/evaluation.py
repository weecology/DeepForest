import os
import glob
import cv2
import random
import numpy as np
import rasterio

import slidingwindow as sw
import fiona

#Plotting and polygon overlap
from shapely.ops import cascaded_union
from shapely.geometry import box
from shapely.geometry import shape
from rtree import index
from scipy.optimize import linear_sum_assignment
from itertools import chain
from PIL import Image

#NEON recall rate
import pandas as pd
from shapely.geometry import Point
from matplotlib import pyplot

from keras_retinanet.utils.visualization import draw_detections, draw_annotations, draw_ground_overlap

def neonRecall(
    site,
    generator,
    model,
    score_threshold=0.05,
    max_detections=100,
    suppression_threshold=0.2,
    save_path=None,
    experiment=None,
    DeepForest_config = None):

    #load field data
    field_data=pd.read_csv("data/field_data.csv") 
    field_data=field_data[field_data['UTM_E'].notnull()]

    #select site
    site_data=field_data[field_data["siteID"]==site]

    #select tree species
    specieslist=pd.read_csv("data/AcceptedSpecies.csv")
    specieslist =  specieslist[specieslist["siteID"]==site]

    site_data=site_data[site_data["scientificName"].isin(specieslist["scientificName"].values)]

    #Single bole individuals as representitve, no individualID ending in non-digits
    site_data=site_data[site_data["individualID"].str.contains("\d$")]

    #Only data within the last two years, sites can be hand managed
    #site_data=site_data[site_data["eventID"].str.contains("2015|2016|2017|2018")]

    #Get remaining plots
    plots=site_data.plotID.unique()

    point_contains=[]
    for plot in plots:

        #select plot
        plot_data=site_data[site_data["plotID"]==plot]

        #load plot image
        tile="data/" + site + "/" + plot + ".tif"
        
        #Check if file exists, if not, skip
        if os.path.exists(tile):
            numpy_image=load_image(tile)
        else:
            continue

        #Gather detections
        final_boxes=predict_tile(numpy_image,generator,model,score_threshold,max_detections,suppression_threshold)            

        #If empty, skip.
        if final_boxes is None:
            continue
    
        #Find geographic bounds
        with rasterio.open(tile) as dataset:
            bounds=dataset.bounds   
    
        #Save image and send it to logger
        if save_path is not None:
            draw_detections(numpy_image, final_boxes[:,:4], final_boxes[:,4], final_boxes[:,5], label_to_name=generator.label_to_name,score_threshold=0.05)
    
            #add points
            x=(plot_data.UTM_E- bounds.left).values/0.1
            y=(bounds.top - plot_data.UTM_N).values/0.1
            for i in np.arange(len(x)):
                cv2.circle(numpy_image,(int(x[i]),int(y[i])), 5, (0,0,255), 1)
    
            cv2.imwrite(os.path.join(save_path, '{}_NeonPlot.png'.format(plot)), numpy_image)
            if experiment:
                experiment.log_image(os.path.join(save_path, '{}_NeonPlot.png'.format(plot)),file_name=str(plot))            
    
        projected_boxes = []
        for row in  final_boxes:
            #Add utm bounds and create a shapely polygon
            pbox=create_polygon(row, bounds,cell_size=0.1)
            projected_boxes.append(pbox)
    
        #for each point
    
        for index,tree in plot_data.iterrows():
            p=Point(tree.UTM_E,tree.UTM_N)
    
            within_polygon=[]
    
            for prediction in projected_boxes:
                within_polygon.append(p.within(prediction))
    
            #Check for overlapping polygon, add it to list
            point_contains.append(sum(within_polygon) > 0)

    if len(point_contains)==0:
        recall=0
    else:
        ## Recall rate for plot
        recall=sum(point_contains)/len(point_contains)

    #Log
    return(recall)

#Processing Functions

#IoU for non-rectangular polygons
def compute_windows(numpy_image,pixels=400,overlap=0.05):
    windows = sw.generate(numpy_image, sw.DimOrder.HeightWidthChannel, pixels,overlap )
    return(windows)

def retrieve_window(numpy_image,window):
    crop=numpy_image[window.indices()]
    return(crop)

def load_image(tile):
    im = Image.open(tile)
    numpy_image = np.array(im)

    #BGR order
    numpy_image=numpy_image[:,:,::-1].copy()
    return(numpy_image)

def non_max_suppression(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes	
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return the indices for only the bounding boxes that were picked using the
    # integer data type
    return pick


def predict_tile(numpy_image,generator,model,score_threshold,max_detections,suppression_threshold):
    #get sliding windows
    windows=compute_windows(numpy_image)

    #holder for all detections among windows within tile
    plot_detections=[]

    #Prediction for each window
    for window in windows:
        raw_image=retrieve_window(numpy_image,window)

        #utilize the generator to scale?
        image, scale = generator.resize_image(raw_image)

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

        #Stop if no predictions
        if len(image_detections)==0:
            continue

        #align detections to original image
        x,y,w,h=window.getRect()

        #boxes are in form x1, y1, x2, y2
        image_detections[:,0] = image_detections[:,0] + x 
        image_detections[:,1] = image_detections[:,1] + y 
        image_detections[:,2] = image_detections[:,2] + x 
        image_detections[:,3] = image_detections[:,3] + y 

        #Collect detection across windows
        plot_detections.append(image_detections)                

    #If no predictions in any window
    if len(plot_detections)==0:
        return None

    #Non-max supression
    all_boxes=np.concatenate(plot_detections)
    final_box_index=non_max_suppression(all_boxes[:,:4], overlapThresh=suppression_threshold)
    final_boxes=all_boxes[final_box_index,:]
    return final_boxes

def create_polygon(row,bounds,cell_size):

    #boxes are in form x1, y1, x2, y2, add the origin utm extent
    x1= (row[0]*cell_size) + bounds.left
    y1 = bounds.top - (row[1]*cell_size) 
    x2 =(row[2]*cell_size) + bounds.left
    y2 = bounds.top - (row[3]*cell_size) 

    b = box(x1, y1, x2, y2)

    return(b)

def IoU_polygon(a,b):

    #Area of predicted box
    predicted_area=b.area

    #Area of ground truth polygon
    polygon_area=a.area

    #Intersection
    intersection_area=a.intersection(b).area

    iou = intersection_area / float(predicted_area + polygon_area - intersection_area)

    return iou

def calculateIoU(itcs,predictions):

    '''
    1) Find overlap among polygons efficiently 
    2) Calulate a cost matrix of overlap, with rows as itcs and columns as predictions
    3) Hungarian matching for pairing
    4) Calculate intersection over union (IoU)
    5) Mean IoU returned.
    '''
    # Populate R-tree index with bounds of prediction boxes
    idx = index.Index()

    for pos, cell in enumerate(predictions):
        # assuming cell is a shapely object
        idx.insert(pos, cell.bounds)

    #Create polygons
    itc_polygons=[shape(x["geometry"]) for x in itcs["data"]]

    overlap_dict={}

    #select predictions that overlap with the polygons
    matched=[predictions[x] for x in idx.intersection(itcs["bounds"])]

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

    #Jaccard evaluation
def Jaccard(
        generator,
        model,
        iou_threshold=0.5,
        score_threshold=0.05,
        max_detections=100,
        suppression_threshold=0.2,
        save_path=None,
        experiment=None,
        DeepForest_config = None
        ):
        """ Evaluate a given dataset using a given model.

        # Arguments
            generator       : The generator that represents the dataset to evaluate.
            model           : The model to evaluate.
            iou_threshold   : The threshold used to consider when a detection is positive or negative.
            score_threshold : The score confidence threshold to use for detections.
            max_detections  : The maximum number of detections to use per image.
            save_path       : The path to save images with visualized detections to.
            experiment     : Comet ml experiment to evaluate
        # Returns
            A dict mapping class names to mAP scores.
        """

        #Load ground truth polygons
        ground_truth, ground_truth_tiles, ground_truth_utmbox=_load_groundtruth(DeepForest_config)

        plot_IoU ={}

        for plot in ground_truth:

            print(plot)

            #Load polygons
            polys=ground_truth[plot]["data"]

            #read rgb tile
            tile=ground_truth_tiles[plot]
            numpy_image=load_image(tile)

            #Gather detections
            final_boxes=predict_tile(numpy_image,
                                     generator,
                                     model,
                                     score_threshold,
                                     max_detections,
                                     suppression_threshold)

            #Save image and send it to logger
            if save_path is not None:
                draw_detections(numpy_image, final_boxes[:,:4], final_boxes[:,4], final_boxes[:,5], label_to_name=generator.label_to_name,score_threshold=0.05)
                cv2.imwrite(os.path.join(save_path, '{}.png'.format(plot)), numpy_image)
                if experiment:              
                    experiment.log_image(os.path.join(save_path, '{}.png'.format(plot)),file_name=str(plot)+"groundtruth")

            #Find overlap 
            projected_boxes=[]

            for row in  final_boxes:

                #Add utm bounds and create a shapely polygon
                pbox=create_polygon(row, ground_truth_utmbox[plot],cell_size=0.1)
                projected_boxes.append(pbox)

            if save_path is not None:
                draw_ground_overlap(plot,ground_truth,ground_truth_tiles,projected_boxes,save_path=save_path,experiment=experiment)

            #Match overlap and generate cost matrix and fill with non-zero elements
            IoU=calculateIoU(ground_truth[plot], projected_boxes)

            plot_IoU[plot]=IoU

        #Mean IoU across all plots
        #Mean IoU across all plots
        all_values=list(plot_IoU.values())
        meanIoU=np.mean(list(chain(*all_values)))
        return meanIoU

#load ground truth polygons and tiles
def _load_groundtruth(DeepForest_config):

    #Returns ground truth polygons, path to tif files, and bounding boxes    
    ground_truth={}

    shps=glob.glob(DeepForest_config["itc_path"],recursive=True)

    for shp in shps:

        items = {}

        #Read polygons
        with fiona.open(shp,"r") as source:

            items["data"] = list(source)
            items["bounds"] = source.bounds
            #Label by plot ID, all records are from the same plot
            ground_truth[source[0]["properties"]["Plot_ID"]]=items

    #Corresponding tiles
    ground_truth_tiles={}

    for plot in ground_truth:
        ground_truth_tiles[plot]= DeepForest_config["itc_tile_path"] + plot + ".tif"

    #Find extent of each tile for projection
    ground_truth_utmbox = {}

    #Holder for bad indices to remove.
    to_remove=[]

    for plot in ground_truth:

        if not os.path.exists(ground_truth_tiles[plot]): # check first if file exsits
            print("missing tile %s, removing from list" % ground_truth_tiles[plot])
            to_remove.append(plot)
            continue

        with rasterio.open(ground_truth_tiles[plot]) as dataset:
            ground_truth_utmbox[plot]=dataset.bounds            

    #Drop missing tile
    for key in to_remove:
        del ground_truth[key]

    return(ground_truth,
           ground_truth_tiles,
           ground_truth_utmbox
           )