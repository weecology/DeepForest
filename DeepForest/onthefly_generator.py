"""
On the fly self. Crop out portions of a large image, and pass boxes and annotations. This follows the csv_self template. Satifies the format in self.py
"""
import keras
import tensorflow as tf
import random
import itertools
import csv
import sys
import os.path
import cv2
import rasterio
import pandas as pd
import numpy as np
from PIL import Image
from six import raise_from
import slidingwindow as sw
from DeepForest import Lidar, postprocessing, utils
from matplotlib import pyplot
from keras_retinanet.preprocessing.generator import Generator
from keras_retinanet.utils.image import preprocess_image, resize_image
from keras_retinanet import models

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
        group_method="none",
        name=None,
        **kwargs
    ):
        """ Initialize a data self.

        """
        #Assign config and intiliaze values
        self.DeepForest_config=DeepForest_config
        self.rgb_res=DeepForest_config['rgb_res']        
        self.image_names = []
        self.image_data  = {}
        self.name = name
        self.windowdf = windowdf
        self.batch_size = DeepForest_config["batch_size"]
        self.group_order = {}
        self.group_method=group_method
        self.shuffle_tile_epoch=shuffle_tile_epoch
        self.annotation_list = data  
        
        #Tensorflow prediction session
        self.session_exists = False
        
        #Set destination directory
        if not base_dir:
            self.base_dir=DeepForest_config["rgb_tile_dir"]
            
        #Holder for image path, keep from reloading same image to save time.
        self.previous_image_path = None
        
        #Holder for previous annotations, after epoch > 1
        self.annotation_dict={}
    
        #Compute sliding windows, assumed that all objects are the same extent and resolution
        self.windows = self.compute_windows()
        
        #Read classes
        self.classes=self.read_classes()  
        
        #Create label dict
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key                
        
        super(OnTheFlyGenerator, self).__init__(**kwargs)
                        
    def __len__(self):
        """Number of batches for self"""
        return len(self.groups)
         
    def size(self):
        """ Size of the dataset.
        """
        image_data= self.windowdf.to_dict("index")
        image_names = list(image_data.keys())
        
        return len(image_names)
    
    def define_groups(self, shuffle=False):
        
        '''
        Define image data and names based on grouping of tiles for computational efficiency 
        '''
        #group by tile
        groups = [df for _, df in self.windowdf.groupby('tile')]
        
        if shuffle:
            #Shuffle order of windows within a tile
            groups=[x.sample(frac=1) for x in groups]      
            
            #Shuffle order of tiles
            random.shuffle(groups)
        
        #Bring back together
        newdf = pd.concat(groups).reset_index(drop=True)
        
        image_data = newdf.to_dict("index")
        image_names = list(image_data.keys())
        
        return(image_data, image_names)
    
    def read_classes(self):
        """ 
        Number of annotation classes
        """
        
        #Get unique classes
        uclasses=self.annotation_list.loc[:,['label','numeric_label']].drop_duplicates()
        
        # Define classes 
        classes = {}
        for index, row in uclasses.iterrows():
            classes[row.label] = row.numeric_label
        
        return(classes)
    
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
    
    def compute_windows(self):
        ''''
        Create a sliding window object
        '''
        #Load tile
        image = self.base_dir + self.annotation_list.rgb_path.unique()[0]
        im = Image.open(image)
        numpy_image = np.array(im)    
        
        #Generate sliding windows
        windows = sw.generate(numpy_image, sw.DimOrder.HeightWidthChannel,  self.DeepForest_config["patch_size"], self.DeepForest_config["patch_overlap"])
        
        return(windows)
    
    def retrieve_window(self):
        """
        #Get image crop from tile and window index
        """
        index = self.row["window"]
        crop = self.numpy_image[self.windows[index].indices()]
        return(crop)
    
    def clip_las(self):
        '''' Inherit LIDAR methods for Class
        '''
        self.clipped_las = Lidar.clip_las(lidar_tile=self.lidar_tile, annotations=self.annotation_list, row=self.row, windows=self.windows, rgb_res=self.rgb_res)
        return self.clipped_las
    
    def fetch_lidar_filename(self):           
        lidar_filepath=Lidar.fetch_lidar_filename(self.row, self.DeepForest_config["lidar_path"], self.DeepForest_config["evaluation_site"])
        
        return lidar_filepath

    def load_lidar_tile(self):
        '''Load a point cloud into memory from file
        '''
        self.lidar_filepath=self.fetch_lidar_filename()        
        self.lidar_tile=Lidar.load_lidar(self.lidar_filepath)
        
        return self.lidar_tile
    
    def load_rgb_tile(self):
        ''''
        Read RGB tile from file
        '''
        im = Image.open(self.base_dir+self.row["tile"])
        numpy_image = np.array(im)    
        
        return numpy_image
        
    def compute_CHM(self):
        '''' Compute a canopy height model on loaded point cloud
        '''
        CHM = Lidar.compute_chm(self.clipped_las, kernel_size=self.DeepForest_config["kernel_size"])
        return CHM
    
    def bind_array(self):
        """ Bind RGB and LIDAR arrays
        """
        four_channel_image=Lidar.bind_array(self.image, self.CHM.array) 
        return four_channel_image
    
    def load_new_crop(self):
        ''''Read a new pair of RGB and LIDAR crop
        '''
        #Load rgb image and get crop
        image = self.retrieve_window()

        #BGR order
        image = image[:,:,::-1]
        
        #Store if needed for show, in RGB color space.
        self.image = image        
        
        #Save image path for next evaluation to check
        self.previous_image_path = self.row["tile"]
        
        return image
    
    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        #Select sliding window and tile
        image_name = self.image_names[image_index]        
        self.row = self.image_data[image_name]
        
        ##Check if image the is same as previous draw from self
        if not self.row["tile"] == self.previous_image_path:
            
            print("Loading new tile {}".format(self.row["tile"]))
            
            self.numpy_image = self.load_rgb_tile()
            #self.lidar_tile = self.load_lidar_tile()
            
        #Load a new crop from self
        three_channel_image = self.load_new_crop()
        
        if three_channel_image is None:
            return None
        else: 
            return three_channel_image
    
    def fetch_annotations(self):
        '''
        Find annotations that match the sliding window.
        Note that the window method is calculated once in train.py, this assumes all tiles have the same size and resolution
        offset: Number of meters to add to box edge to look for annotations
        '''
        image = self.base_dir + self.row["tile"]
        index = self.row["window"]
        annotations = self.annotation_list
        windows = self.windows
        offset = (self.DeepForest_config["patch_size"] * 0.1)/self.rgb_res
        patch_size = self.DeepForest_config["patch_size"]
    
        #Find index of crop and create coordinate box
        x, y, w, h=windows[index].getRect()
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
        
        overlapping_boxes=d[d.apply(utils.box_overlap, window=window_coords, axis=1) > 0.5].copy()
        
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
    
    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        #Find the original data and crop
        image_name = self.image_names[image_index]
        self.row = self.image_data[image_name]
        
        #Check for blank black image, if so, enforce no annotations
        remove_annotations = utils.image_is_blank(self.image)
        
        if remove_annotations:
            return np.zeros((0, 5))
        
        #Look for annotations in previous epoch
        key = self.row["tile"]+"_"+str(self.row["window"])
        
        if key in self.annotation_dict:
            boxes=self.annotation_dict[key]
        else:
            #Which annotations fall into that crop?
            self.annotation_dict[key] = self.fetch_annotations()

        #Index
        boxes=np.copy(self.annotation_dict[key])
        
        #Convert to float if needed
        if not boxes.dtype==np.float64:   
            boxes=boxes.astype("float64")
        
        return boxes

    def utm_from_window(self):
        """Given the current window crop, get the utm position from the rasterio metadata
        returns: utm bounds"""
        x,y,h,w = self.windows[self.row["window"]].getRect()
        x = x * self.rgb_res
        y = y * self.rgb_res
        
        with rasterio.open(self.base_dir+self.row["tile"]) as dataset:
            self.utm_bounds = dataset.bounds   
        
        utm_xmin = self.utm_bounds.left + x
        utm_xmax = self.utm_bounds.left + x + (self.DeepForest_config["patch_size"] * self.rgb_res)
        
        utm_ymax = self.utm_bounds.top - y
        utm_ymin = self.utm_bounds.top - y - (self.DeepForest_config["patch_size"] * self.rgb_res)
        
        return (utm_xmin, utm_xmax, utm_ymin, utm_ymax)
   
    #Prediction methods
    def get_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)
    
    def tf_session(self):
        # set the modified tf session as backend in keras        
        keras.backend.tensorflow_backend.set_session(self.get_session())        
        
        #flag for session setting
        self.session_exists = True
        
    def predict(self, model):
        '''
        Generic prediction function for bounding box prediction
        model: a loaded model of form retinanet saved model
        '''
                
        #Get a new tensorflow session if not previously set 
        if not self.session_exists:
            self.tf_session()
            
        # load retinanet model
        model = models.load_model(model, backbone_name='resnet50', convert=True, nms_threshold=self.DeepForest_config["nms_threshold"])
        labels_to_names = {0: 'Tree'}
                   
        # preprocess image for network
        image = preprocess_image(self.image)
        image, scale = resize_image(img = image, min_side = 400)
        
        # process image
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        
        # correct for image scale
        boxes /= scale
            
        #only pass score threshold boxes
        quality_boxes = []
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            quality_boxes.append(box)
            # scores are sorted so we can break
            if score < self.DeepForest_config["score_threshold"]:
                break
        
        ###Drape boxes on lidar cloud###
        #Load tile
        self.load_lidar_tile()
        
        #The tile could be the full tile, so let's check just the 400 pixel crop we are interested    
        density = Lidar.check_density(self.lidar_tile)
    
        print("Point density is {:.2f}".format(density))
    
        if density > 4:
            #find window utm coordinates
            pc = postprocessing.drape_boxes(boxes=quality_boxes, pc = self.lidar_tile)     
    
            #Get new bounding boxes
            new_boxes = postprocessing.cloud_to_box(pc)    
            new_scores = scores[:new_boxes.shape[0]]
            new_labels = labels[:new_boxes.shape[0]]          
        else:
            print("Point density of {:.2f} is too low, skipping image {}".format(density, self.row["tile"]))        
                   
        return [new_boxes, new_scores, new_labels]       
        
        