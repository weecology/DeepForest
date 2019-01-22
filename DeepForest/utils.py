"""
Utility functions for reading and cleaning images
"""

#Utility functions

def normalize(image):
    
    #Range normalize for all channels 
    
    v_min = image.min(axis=(0, 1), keepdims=True)
    v_max = image.max(axis=(0, 1), keepdims=True)
    normalized_image=(v - v_min)/(v_max - v_min)
    
    return normalized_image

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