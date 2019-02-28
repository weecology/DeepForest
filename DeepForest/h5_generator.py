"""
On the fly generator. Crop out portions of a large image, and pass boxes and annotations. This follows the csv_generator template. Satifies the format in generator.py
"""
import pandas as pd
import h5py

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

from DeepForest import Lidar

class H5Generator(Generator):
    """ Generate data for a custom CSV dataset.

    See https://github.com/fizyr/keras-retinanet#csv-datasets for more information.
    """

    def __init__(
        self,
        data,
        DeepForest_config,
        h5_dir,        
        base_dir=None,
        group_method="none",
        name=None,
        **kwargs
    ):
        """ Initialize a data generator.

        """
        self.image_names = []
        self.image_data  = {}
        self.name=name
        self.windowdf=data
        self.DeepForest_config = DeepForest_config
        self.site=DeepForest_config["evaluation_site"]
        self.h5_dir = h5_dir
        
        #Holder for the group order, after shuffling we can still recover loss -> window
        self.group_order = {}
        self.group_method=group_method
        
        #Define base directory        
        if not base_dir:
            self.base_dir = DeepForest_config["rgb_tile_dir"]
        else:
            self.base_dir = base_dir
            
        #Holder for image path, keep from reloading same image to save time.
        self.previous_image_path=None
        
        #Read classes
        self.classes={"Tree": 0}
        
        #Create label dict
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key        
        
        #Set groups at first order.
        self.define_groups(shuffle=False)
        
        #report total number of annotations
        self.total_trees=self.total_annotations()
                
        super(H5Generator, self).__init__(**kwargs)
                        
    def __len__(self):
        """Number of batches for generator"""
        return len(self.groups)
         
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

    def total_annotations(self):
        """ Find the total number of annotations for the dataset
        """
        #Find matching annotations        
        tiles=self.windowdf["tile"].unique()
        
        total_annotations = 0
                
        #Select annotations
        for tilename in tiles:
            csv_name = os.path.join(self.h5_dir, os.path.splitext(tilename)[0]+'.csv')
            try:
                annotations = pd.read_csv(csv_name)
            except Exception as e:
                print(e)
                print("The csv named {} from tilename {} encountered an error".format(csv_name, tilename))
                
            selected_annotations = pd.merge(self.windowdf, annotations)
            total_annotations += len(selected_annotations)        
        
        print("There are a total of {} tree annotations in the {} generator".format(total_annotations,self.name))       
        
        return(total_annotations)
    
    def define_groups(self, shuffle=False):
        '''
        Define image data and names based on grouping of tiles for computational efficiency 
        '''
        #group by tile
        groups = [df for _, df in self.windowdf.groupby('tile')]
        
        if shuffle:
            #Shuffle order of windows within a tile
            groups = [x.sample(frac=1) for x in groups]      
            
            #Shuffle order of tiles
            random.shuffle(groups)
        
        #Bring pandas frame back together
        newdf=pd.concat(groups).reset_index(drop=True)
        image_data=newdf.to_dict("index")
        image_names = list(image_data.keys())
        
        return(image_data, image_names)
    
    def fetch_lidar_filename(self):           
        lidar_filepath=Lidar.fetch_lidar_filename(self.row, self.base_dir)
        
        if lidar_filepath:
            return lidar_filepath
        else:
            print("Lidar file {} cannot be found in {}".format(self.row["tile"], self.base_dir))
            raise IOError 
        
    def load_lidar_tile(self):
        '''Load a point cloud into memory from file
        '''
        self.lidar_filepath=self.fetch_lidar_filename()        
        self.lidar_tile=Lidar.load_lidar(self.lidar_filepath)
    
    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        #Select sliding window and tile
        try:
            image_name = self.image_names[image_index]
        except Exception as e:
            print("Failed on image index {}".format(image_index))
            print("There are {} names in the image names object".format(len(self.image_names)))
        
        self.row = self.image_data[image_name]
        
        #Open image to crop
        ##Check if tile the is same as previous draw from generator, this will save time.
        if not self.row["tile"] == self.previous_image_path:
            
            print("Loading new tile: %s" %(self.row["tile"]))
            
            #tilename for h5 and csv files
            tilename = os.path.split(self.row["tile"])[-1]
            tilename = os.path.splitext(tilename)[0]                        
            
            h5_name = os.path.join(self.h5_dir, tilename+'.h5')
            csv_name = os.path.join(self.h5_dir, tilename+'.csv')
            
            #Read h5 
            self.hf = h5py.File(h5_name, 'r')
            train_addrs = self.hf['train_imgs']
            
            #Read corresponding csv labels
            self.annotations = pd.read_csv(csv_name)
            
        #read image from h5
        window = self.row["window"]
        image = self.hf["train_imgs"][window,...]
        
        #Store RGB if needed for show, in RGB color space.
        self.image = image[:,:,:3]        
        
        #Save image path for next evaluation to check
        self.previous_image_path = self.row["tile"]
            
        return image
    
    def load_annotations(self, image_index):
        '''
        Load annotations from csv file
        '''
        #Select sliding window and tile
        image_name = self.image_names[image_index]        
        self.row = self.image_data[image_name]
       
        #Find annotations
        annotations = self.annotations.loc[(self.annotations["tile"] == self.row["tile"]) & (self.annotations["window"] == self.row["window"])]
            
        return annotations[["0","1","2","3","4"]].values
    
    def compute_windows(self):
        ''''
        Create a sliding window object for reference
        '''
        #Load tile
        image = self.base_dir + self.annotation_list.rgb_path.unique()[0]
        im = Image.open(image)
        numpy_image = np.array(im)    
        
        #Generate sliding windows
        windows = sw.generate(numpy_image, sw.DimOrder.HeightWidthChannel,  self.DeepForest_config["patch_size"], self.DeepForest_config["patch_overlap"])
        
        return(windows)
    
if __name__=="__main__":
    
    import yaml
    import preprocess
    
    #load config
    with open('../_config_debug.yml', 'r') as f:
        DeepForest_config = yaml.load(f)    
    
    #Load data
    data=preprocess.load_csvs(h5_dir=DeepForest_config["training_h5_dir"])
    
    #Split training and test data
    train, test = preprocess.split_training(data, DeepForest_config, experiment=None)
    
    generator=H5Generator(train, DeepForest_config)
    
    for i in range(generator.size()):
        image=generator.load_image(i)
        print(image.shape)
        