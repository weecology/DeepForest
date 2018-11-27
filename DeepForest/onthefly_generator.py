"""
On the fly generator. Crop out portions of a large image, and pass boxes and annotations. This follows the csv_generator template. Satifies the format in generator.py
"""
import pandas as pd

from keras_retinanet.preprocessing.generator import Generator
from keras_retinanet.utils.image import read_image_bgr
from keras_retinanet.utils.visualization import draw_annotations

import numpy as np
from PIL import Image
from six import raise_from
import random

import csv
import sys
import os.path

import cv2
import slidingwindow as sw
import itertools

from DeepForest.lidar_crop import compute_chm, pad_array, fetch_lidar_tile, bind_array
from matplotlib import pyplot


class OnTheFlyGenerator(Generator):
    """ Generate data for a custom CSV dataset.

    See https://github.com/fizyr/keras-retinanet#csv-datasets for more information.
    """

    def __init__(
        self,
        data,
        windowdf,
        DeepForest_config,
        base_dir=None,
        shuffle_tile_epoch=False,
        name=None,
        **kwargs
    ):
        """ Initialize a CSV data generator.

        """
        self.image_names = []
        self.image_data  = {}
        self.name=name
        self.windowdf=windowdf
        self.lidar_path=DeepForest_config["lidar_path"]
        
        #Holder for the group order, after shuffling we can still recover loss -> window
        self.group_order = {}
        
        #Shuffle at the end of an epoch
        self.shuffle_tile_epoch=shuffle_tile_epoch
        
        #Store DeepForest_config and resolution
        self.DeepForest_config=DeepForest_config
        self.rgb_res=DeepForest_config['rgb_res']
        
        if not base_dir:
            self.base_dir=DeepForest_config["rgb_tile_dir"]
        else:
            self.base_dir = base_dir + "/"
            
        #Holder for image path, keep from reloading same image to save time.
        self.previous_image_path=None
        
        #Holder for previous annotations, after epoch > 1
        self.annotation_dict={}
        
        #Read annotations into pandas dataframe
        self.annotation_list=data  

        #Compute sliding windows, assumed that all objects are the same extent and resolution
        self.windows=compute_windows( self.base_dir+ self.annotation_list.rgb_path.unique()[0], DeepForest_config["patch_size"], DeepForest_config["patch_overlap"])
        
        #Read classes
        self.classes=_read_classes(data=self.annotation_list)  
        
        #Create label dict
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key        
        
        #Create list of sliding windows to select        
        self.image_data, self.image_names =self.define_groups(self.windowdf,shuffle=False)
        
        super(OnTheFlyGenerator, self).__init__(**kwargs)
                        
    def size(self):
        """ Size of the dataset.
        """
   
        image_data= self.windowdf.to_dict("index")
        image_names = list(image_data.keys())
        
        return len(image_names)

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        return max(self.classes.values()) + 1

    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.labels[label]
    
    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        # PIL is fast for metadata
        image = Image.open(self.image_path(image_index))
        return float(image.width) / float(image.height)

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        #Select sliding window and tile
        image_name=self.image_names[image_index]        
        row=self.image_data[image_name]
                
        #Open image to crop
        ##Check if image the is same as previous draw from generator, this will save time.
        if not row["image"] == self.previous_image_path:
            print("Loading new tile: %s" %(row["image"]))
            im = Image.open(self.base_dir+row["image"])
            
            #Read numpy array for RGB
            self.numpy_image = np.array(im)    
            
            #Finding the corresponding lidar tile
            self.lidar_tile=fetch_lidar_tile(row,self.lidar_path)
        
        #Load rgb image and get crop
        image=retrieve_window(numpy_image=self.numpy_image,index=row["windows"],windows=self.windows)

        #BGR order
        image=image[:,:,::-1].copy()
        
        #Store if needed for show, in RGB color space.
        self.image=image        
        
        #Save image path for next evaluation to check
        self.previous_image_path = row["image"]
        
        #If not lidar, skip
        if self.lidar_tile == None:
            return None        
        
        #LIDAR CHM
        CHM=compute_chm(lidar_tile=self.lidar_tile,annotations=self.annotation_list, row=row, windows=self.windows, rgb_res=self.rgb_res)
        
        #Bind RGB and LIDAR arrays
        four_channel_image=bind_array(image,CHM.array)
            
        return four_channel_image

    def define_groups(self,windowdf,shuffle=False):
        
        '''
        Define image data and names based on grouping of tiles for computational efficiency 
        '''
        
        #group by tile
        groups = [df for _, df in windowdf.groupby('image')]
        
        if shuffle:
            #Shuffle order of windows within a tile
            groups=[x.sample(frac=1) for x in groups]      
            
            #Shuffle order of tiles
            random.shuffle(groups)
        
        #Bring back together
        newdf=pd.concat(groups).reset_index(drop=True)
        
        image_data=newdf.to_dict("index")
        image_names = list(image_data.keys())
        
        return(image_data,image_names)
    
    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        
        #Find the original data and crop
        image_name=self.image_names[image_index]
        row=self.image_data[image_name]
        
        #Check for blank image, if so, enforce no annotations
        remove_annotations=image_is_blank(self.image)
        
        if remove_annotations:
            return np.zeros((0, 5))
        
        #Look for annotations in previous epoch
        key=row["image"]+"_"+str(row["windows"])
        
        if key in self.annotation_dict:
            boxes=self.annotation_dict[key]
        else:
            #Which annotations fall into that crop?
            self.annotation_dict[key]=fetch_annotations(image=self.base_dir+row["image"],
                                           index=row["windows"],
                                           annotations=self.annotation_list,
                                           windows=self.windows,
                                           offset=(self.DeepForest_config["patch_size"] * 0.1)/self.rgb_res,
                                           patch_size=self.DeepForest_config["patch_size"])

        #Index
        boxes=np.copy(self.annotation_dict[key])
        
        #Convert to float if needed
        if not boxes.dtype==np.float64:   
            boxes=boxes.astype("float64")
        
        return boxes
    
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

#Find window indices
def compute_windows(image,pixels=250,overlap=0.05):
    im = Image.open(image)
    numpy_image = np.array(im)    
    windows = sw.generate(numpy_image, sw.DimOrder.HeightWidthChannel, pixels,overlap )
    return(windows)

#Get image from tile and window index
def retrieve_window(numpy_image,index,windows):
    crop=numpy_image[windows[index].indices()]
    return(crop)

def _read_classes(data):
    """ 
    """
    
    #Get unique classes
    uclasses=data.loc[:,['label','numeric_label']].drop_duplicates()
    
    # Define classes 
    classes = {}
    for index, row in uclasses.iterrows():
        classes[row.label] = row.numeric_label
    
    return(classes)
    
def fetch_annotations(image,index,annotations,windows,offset,patch_size):
    '''
    Find annotations that match the sliding window.
    Note that the window method is calculated once in train.py, this assumes all tiles have the same size and resolution
    offset: Number of meters to add to box edge to look for annotations
    '''

    #Find index of crop and create coordinate box
    x,y,w,h=windows[index].getRect()
    
    window_coords={}

    #top left
    window_coords["x1"]=x
    window_coords["y1"]=y
    
    #Bottom right
    window_coords["x2"]=x+w    
    window_coords["y2"]=y+h    
    
    #convert coordinates such that box is shown with respect to crop origin
    annotations["window_xmin"] = annotations["origin_xmin"]- window_coords["x1"]
    annotations["window_ymin"] = annotations["origin_ymin"]- window_coords["y1"]
    annotations["window_xmax"] = annotations["origin_xmax"]- window_coords["x1"]
    annotations["window_ymax"] = annotations["origin_ymax"]- window_coords["y1"]

    #Quickly subset a reasonable set of annotations based on sliding window
    d=annotations[ 
        (annotations["rgb_path"]==image.split("/")[-1]) &
        (annotations.window_xmin > -offset) &  
        (annotations.window_ymin > -offset)  &
        (annotations.window_xmax < (patch_size+ offset)) &
        (annotations.window_ymax < (patch_size+ offset))
                     ]
    
    overlapping_boxes=d[d.apply(box_overlap,window=window_coords,axis=1) > 0.5]
    
    #If boxes fall off edge, clip to window extent    
    overlapping_boxes.loc[overlapping_boxes["window_xmin"] < 0,"window_xmin"]=0
    overlapping_boxes.loc[overlapping_boxes["window_ymin"] < 0,"window_ymin"]=0
    
    #The max size depends on the sliding window
    max_height=window_coords['y2']-window_coords['y1']
    max_width=window_coords['x2']-window_coords['x1']
    
    overlapping_boxes.loc[overlapping_boxes["window_xmax"] > max_width,"window_xmax"]=max_width
    overlapping_boxes.loc[overlapping_boxes["window_ymax"] > max_height,"window_ymax"]=max_height
    
    #format
    boxes=overlapping_boxes[["window_xmin","window_ymin","window_xmax","window_ymax","numeric_label"]].values
    
    return(boxes)    


def box_overlap(row,window):
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

