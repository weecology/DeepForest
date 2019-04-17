import os
import glob
import cv2
import random
import numpy as np
import rasterio
import slidingwindow as sw

#Plotting and polygon overlap
from shapely.geometry import box, shape, Point
from rtree import index
from PIL import Image
import geopandas as gp

#NEON recall rate
import pandas as pd
from matplotlib import pyplot

from keras_retinanet.utils.visualization import draw_detections, draw_annotations
from keras_retinanet.utils.eval import _get_detections

#DeepForest
from DeepForest import Lidar 
from DeepForest import postprocessing
from DeepForest import onthefly_generator
import copy

def neonRecall(
    sites,
    generator,
    model,
    score_threshold=0.05,
    max_detections=100,
    suppression_threshold=0.15,
    save_path=None,
    experiment=None):

    point_contains = [ ]

    site_data_dict = {}
    for site in sites:
        #Container for recall pts.
        
        #load field data
        field_data = pd.read_csv("data/field_data.csv") 
        field_data = field_data[field_data['UTM_E'].notnull()]
    
        #select site
        site_data = field_data[field_data["siteID"]==site]
    
        #select tree species
        specieslist = pd.read_csv("data/AcceptedSpecies.csv",encoding="utf-8")
        specieslist =  specieslist[specieslist["siteID"] == site]
    
        site_data = site_data[site_data["scientificName"].isin(specieslist["scientificName"].values)]
    
        #Single bole individuals as representitve, no individualID ending in non-digits
        site_data = site_data[site_data["individualID"].str.contains("\d$")]
        site_data_dict[site] = site_data
        
        #Only data within the last two years, sites can be hand managed
        #site_data=site_data[site_data["eventID"].str.contains("2015|2016|2017|2018")]
        
    for i in range(generator.size()):
        
        #Load image
        raw_image    = generator.load_image(i)
        plot_image = copy.deepcopy(raw_image)
   
        #Skip if missing a component data source
        if raw_image is False:
            print("Empty image, skipping")
            continue
        
        #Store plotting images.
        plot_rgb = plot_image[:,:,:3].copy()
        plot_chm = plot_image[:,:,3]     
        
        image        = generator.preprocess_image(raw_image)
        image, scale = generator.resize_image(image)

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
        
        #Find geographic bounds
        base_dir = generator.DeepForest_config[generator.row["site"]][generator.name]["RGB"]
        tile_path = os.path.join(base_dir, generator.image_data[i]["tile"])
        
        with rasterio.open(tile_path) as dataset:
            tile_bounds = dataset.bounds   
    
        #drape boxes
        #get lidar cloud if a new tile, or if not the same tile as previous image.
        if i == 0:
            generator.load_lidar_tile()
        elif not generator.image_data[i]["tile"] == generator.image_data[i-1]["tile"]:
            generator.load_lidar_tile()
        
        #The tile could be the full tile, so let's check just the 400 pixel crop we are interested    
        #Not the best structure, but the on-the-fly generator always has 0 bounds
        if hasattr(generator, 'hf'):
            bounds = generator.hf["utm_coords"][generator.row["window"]]    
        else:
            bounds=[]
            
        density = Lidar.check_density(generator.lidar_tile, bounds=bounds)
        
        #print("Point density is {:.2f}".format(density))
                
        if density > 100:
            #find window utm coordinates
            #print("Bounds for image {}, window {}, are {}".format(generator.row["tile"], generator.row["window"], bounds))
            pc = postprocessing.drape_boxes(boxes=image_boxes, pc = generator.lidar_tile, bounds=bounds)     
            
            #Get new bounding boxes
            new_boxes = postprocessing.cloud_to_box(pc, bounds)    
            new_scores = image_scores[:new_boxes.shape[0]]
            new_labels = image_labels[:new_boxes.shape[0]]          
            image_detections = np.concatenate([new_boxes, np.expand_dims(new_scores, axis=1), np.expand_dims(new_labels, axis=1)], axis=1)
            
        else:
            #print("Point density of {:.2f} is too low, skipping image {}".format(density, generator.row["tile"]))   
            pass

        #add spatial NEON points
        site_data =site_data_dict[generator.row["site"]]
        plotID = os.path.splitext(generator.image_data[i]["tile"])[0]
        plot_data = site_data[site_data.plotID == plotID]

        #Save image and send it to logger
        if save_path is not None:
            
            x = (plot_data.UTM_E - tile_bounds.left).values / 0.1
            y = (tile_bounds.top - plot_data.UTM_N).values / 0.1
            
            for j in np.arange(len(x)):
                cv2.circle(plot_image,(int(x[j]),int(y[j])), 2, (0,0,255), -1)
    
            #Write RGB
            draw_detections(plot_rgb, image_boxes, image_scores, image_labels, label_to_name=generator.label_to_name,score_threshold=score_threshold)
        
            #name image
            image_name=generator.image_names[i]        
            row=generator.image_data[image_name]             
            fname=os.path.splitext(row["tile"])[0] + "_" + str(row["window"])
        
            #Write RGB
            cv2.imwrite(os.path.join(save_path, '{}_NeonPlot.png'.format(fname)), plot_rgb)
            
            plot_chm = plot_chm/plot_chm.max() * 255                        
            chm = np.uint8(plot_chm)
            draw_detections(chm, image_boxes, image_scores, image_labels, label_to_name=generator.label_to_name, score_threshold=score_threshold, color = (80,127,255))            
            cv2.imwrite(os.path.join(save_path, '{}_Lidar_NeonPlot.png'.format(plotID)), chm)
                
            #Format name and save
            if experiment:
                experiment.log_image(os.path.join(save_path, '{}_NeonPlot.png'.format(plotID)),file_name=str(plotID))
                experiment.log_image(os.path.join(save_path, '{}_Lidar_NeonPlot.png'.format(plotID)),file_name=str("Lidar_" + plotID))
        
        #calculate recall
            s = gp.GeoSeries(map(Point, zip(plot_data.UTM_E, plot_data.UTM_N)))
    
        #Calculate recall
        projected_boxes = []
        
        for row in  image_boxes:
            #Add utm bounds and create a shapely polygon
            pbox=create_polygon(row, tile_bounds, cell_size=0.1)
            projected_boxes.append(pbox)
    
        #for each point, is it within a prediction?
        for index, tree in plot_data.iterrows():
            p=Point(tree.UTM_E, tree.UTM_N)
    
            within_polygon=[]
            for prediction in projected_boxes:
                within_polygon.append(p.within(prediction))
    
            #Check for overlapping polygon, add it to list
            is_within = sum(within_polygon) > 0
            point_contains.append(is_within)
                
    #sum recall across plots
    if len(point_contains)==0:
        recall = None
    else:
        ## Recall rate for plot
        recall = sum(point_contains)/len(point_contains)    
        
    return(recall)


#IoU for non-rectangular polygons
def compute_windows(numpy_image, pixels=400, overlap=0.05):
    windows = sw.generate(numpy_image, sw.DimOrder.HeightWidthChannel, pixels,overlap )
    return(windows)

def retrieve_window(numpy_image,window):
    crop=numpy_image[window.indices()]
    return(crop)

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

def create_polygon(row, bounds, cell_size):

    #boxes are in form x1, y1, x2, y2, add the origin utm extent
    x1= (row[0]*cell_size) + bounds.left
    y1 = bounds.top - (row[1]*cell_size) 
    x2 =(row[2]*cell_size) + bounds.left
    y2 = bounds.top - (row[3]*cell_size) 

    b = box(x1, y1, x2, y2)

    return(b)

def IoU_polygon(a, b):

    #Area of predicted box
    predicted_area=b.area

    #Area of ground truth polygon
    polygon_area=a.area

    #Intersection
    intersection_area=a.intersection(b).area
    iou = intersection_area / float(predicted_area + polygon_area - intersection_area)

    return iou

def calculateIoU(itcs, predictions):

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