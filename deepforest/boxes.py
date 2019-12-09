from shapely.geometry import box
import shapely
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

def merge_tiles():
    
    #1) Select one tile's boxes 
    
    #2) Select the boxes in the overlapping buffer
    
    #3) Merge selected boxes with overlapping boxes from all other tiles
    
    #4) Update data frame
    pass

def merge_boxes(target_list,merge_list):
    """Merge boxes that overlap by a threshold
    box_list: a list of shapely bounding box objects  
    threshold:
    """
    
    idx = index.Index()
    
    # Populate R-tree index with bounds of boxes available to be merged
    for pos, bbox in enumerate(merge_list):
        # assuming cell is a shapely object
        idx.insert(pos, bbox.bounds)
    
    # Loop through each Shapely polygon in the target_list
    merged_boxes = []
    for bbox in target_list:
        # Merge cells that have overlapping bounding boxes
        overlapping_boxes = [merge_list[pos] for pos in idx.intersection(bbox.bounds)]
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

    
    
    