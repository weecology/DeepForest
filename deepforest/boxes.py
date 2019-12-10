"""Box module for combining bounding boxes across tile windows"""
import pandas as pd
import numpy as np
from shapely.geometry import box
from shapely.ops import cascaded_union
from rtree import index

def make_box(xmin, ymin, xmax, ymax):
    # box(minx, miny, maxx, maxy)
    bbox = box(xmin, ymin, xmax, ymax)
    
    return bbox

def box_from_df(df):
    """Create a list of shapely boxes from a dataframe
    
    Args:
        df: a pandas dataframe with columns, xmin, ymin, xmax, ymax
        
    Returns:
        box_list: a list of shapely boxes for each prediction
    """
    box_list = [] 
    for index, x in df.iterrows():
        bbox = make_box(x["xmin"],x["ymin"],x["xmax"],x["ymax"])
        box_list.append(bbox)
    
    return box_list

def df_from_box(box_list):
    """Create a pandas dataframe from a list of shapely box objects
    Args:
        box_list: a list of shapely box objects
    Returns:
        df: a pandas dataframe with columns xmin, ymin, xmax, ymax. label is assumed to be "Tree" and score is set to NA
    """
    xmin_list = []
    ymin_list = []
    xmax_list = []
    ymax_list = []
    label_list = []
    score_list = []
    
    for box in box_list:
        xmin, ymin, xmax, ymax = box.bounds
        xmin_list.append(xmin)
        ymin_list.append(ymin)
        xmax_list.append(xmax)
        ymax_list.append(ymax)
        label_list.append("Tree")
        score_list.append(np.NaN)
        
    df = pd.DataFrame({"xmin":xmin_list,"ymin":ymin_list,"xmax":xmax_list,"ymax":ymax_list,"label":label_list,"score":score_list})
    return df
    
def merge_boxes(target_list,pool_list):
    """Merge boxes that overlap by a threshold
    box_list: a list of shapely bounding box objects  
    threshold:
    """
    #Spatial index
    idx = index.Index()
    
    # Populate R-tree index with bounds of boxes available to be merged
    for pos, bbox in enumerate(pool_list):
        # assuming cell is a shapely object
        idx.insert(pos, bbox.bounds)
    
    # Loop through each Shapely polygon in the target_list
    merged_boxes = []
    for bbox in target_list:
        # Merge cells that have overlapping bounding boxes
        overlapping_boxes = [pool_list[pos] for pos in idx.intersection(bbox.bounds)]
        overlapping_boxes.append(bbox)
        merged_box = cascaded_union(overlapping_boxes)
        
        #Skip if empty
        if merged_box.type == "Polygon":
            #get bounding box of the merge
            xmin, ymin, xmax, ymax = merged_box.bounds
            new_box = box(xmin, ymin, xmax, ymax)
            merged_boxes.append(new_box)
        else:
            merged_boxes.append(bbox)
            
    return merged_boxes

    
def merge_tiles(target_df, pool_df, patch_size=400):
    """Given two pandas dataframes (target and pool), create list of box objects and return a new pandas frame of final boxes 
    
    Args:
        target_df: a pandas dataframe with columns xmin, ymin, xmax, ymax. This dataframe are the boxes that will be merged
        pool_df: a pandas dataframe with columns xmin, ymin, xmax, ymax. This dataframe is the pool of boxes to merge with the target_df
        patch_size: pixel distance to search for overlapping annotations
    Returns:
        merged_df: a pandas dataframe of the merged boxes. 
    """
    
    #Reasonable subset of pool dataframe to reduce size
    target_xmin = target_df.xmin.min()
    target_ymin = target_df.ymin.min()
    pool_df = pool_df[((pool_df.xmin - target_xmin) < patch_size) | ((pool_df.ymin - target_ymin) < patch_size) ]
    
    #Create shapely objects
    target_list = box_from_df(target_df)
    pool_list = box_from_df(pool_df)
    
    #Create new boxes
    merged_boxes = merge_boxes(target_list, pool_list)
    
    #Convert back to a pandas frame
    merged_df = df_from_box(merged_boxes)
    
    return merged_df