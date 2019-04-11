"""
Generate training clips and save the images in h5py to reduce clutter
"""
import argparse
import numpy as np
import os
import h5py
import pandas as pd
from . import onthefly_generator, preprocess, config, Lidar
from DeepForest.utils import image_utils
import sys
import glob
import pathlib

#optional suppress warnings
import warnings
warnings.simplefilter("ignore")

def parse_args():    
    
    #Set tile from command line args
    parser = argparse.ArgumentParser(description='Generate crops for training')
    parser.add_argument('--tile', help='filename of the LIDAR tile to process' )
    args = parser.parse_args()    
    
    return args

def run(tile_csv=None, tile_xml = None, mode="train", site=None):
    
    """Crop 4 channel arrays from RGB and LIDAR CHM
    tile_csv: the CSV training file containing the tree detections
    tile_xml: the xml training file for hand annotations (mode==retrain)
    mode: train or retrain. train loads data from the csv files from R, retrain from the xml hand annotations
    """
    
    DeepForest_config = config.load_config()            
    
    if mode == "train":
        lidar_path = DeepForest_config[site]["training"]["LIDAR"]
        data = preprocess.load_data(data_dir=tile_csv, res=0.1, lidar_path=lidar_path)
        
        #Get tile filename for storing
        tilename = data.rgb_path.unique()[0]
        tilename = os.path.splitext(tilename)[0]
            
        #add site index
        data["site"]=site
        
        #Create windows
        base_dir = DeepForest_config[site]["training"]["RGB"]  
        windows = preprocess.create_windows(data, DeepForest_config, base_dir)   
        
        #Check lidar  for point density
        check_lidar = True
        name = "training"
        
        #Destination dir
        destination_dir = DeepForest_config[site]["h5"]
            
    if mode == "retrain":
        #Base dir
        base_dir = DeepForest_config[site]["hand_annotations"]["RGB"]
        
        #Load xml annotations
        data = preprocess.load_xml(path=tile_xml, dirname=base_dir, res=DeepForest_config["rgb_res"])
                    
        data["site"] = site
        tilename = os.path.splitext(os.path.basename(tile_xml))[0] 

        #Create windows
        windows = preprocess.create_windows(data, DeepForest_config, base_dir) 

        #Don't check lidar for density, annotations are made directly on RGB
        check_lidar = True
        name = "hand_annotations"
        
        #destination dir
        destination_dir = os.path.join(DeepForest_config[site]["h5"],"hand_annotations")
    
    #If dest doesn't exist, create it
    if not os.path.exists(destination_dir):
        pathlib.Path(destination_dir).mkdir(parents=True, exist_ok=True)    
        
    if windows is None:
        print("Invalid window, cannot find {} in {}".format(tilename, base_dir))
        return None
    
    #Create generate
    generator = onthefly_generator.OnTheFlyGenerator(data,
                                                     windows,
                                                     DeepForest_config,
                                                     name=name,
                                                     image_min_side=400,
                                                     image_max_side=400
                                                     )
    
    #Create h5 dataset    
    # open a hdf5 file and create arrays
    h5_filename = os.path.join(destination_dir, tilename + ".h5")
    hdf5_file = h5py.File(h5_filename, mode='w')    
    
    #A 3 channel image of square patch size.
    train_shape = (generator.size(), DeepForest_config["patch_size"], DeepForest_config["patch_size"], DeepForest_config["input_channels"])
    
    #Create h5 dataset to fill
    hdf5_file.create_dataset("train_imgs", train_shape, dtype='f')
    
    #Create h5 dataset of utm positions, xmin,xmax,ymin,ymax
    hdf5_file.create_dataset("utm_coords", (generator.size(),4) , dtype='f')
    
    #Generate crops and annotations
    labels = {}
    
    #load first rgb and lidar tile, assumes all images in generator came from same tile, saves time
    generator.load_image(1)
    
    if check_lidar:
        point_cloud = generator.load_lidar_tile()
    
    for i in range(generator.size()):
        
        print("window {i} from tile {tilename}".format(i=i, tilename=tilename))

        #Load images
        image = generator.load_image(i)
        
        #If image window is corrupt (RGB missing), go to next tile, it won't be in labeldf
        if image is None:
            continue
                
        #Check if there is lidar density
        if check_lidar:
            bounds = generator.get_window_extent()
            density = Lidar.check_density(point_cloud, bounds)
                    
            if density < generator.DeepForest_config["min_density"]:
                print("Point density is {} for window {}, skipping".format(density, tilename))
                continue
            
            #Check for a patchy chm, get proportion NA
            propNA = image_utils.proportion_NA(point_cloud)
            if propNA > DeepForest_config["min_coverage"]:
                print("Point density is too patchy ({}%) for window {}, skipping".format(propNA, tilename))
                continue 
        
        #TODO do boxes need to be scaled in some way?    
        #image, scale = generator.resize_image(image)
        
        image = np.expand_dims(image, 2)
        
        hdf5_file["train_imgs"][i,...] = image        
        
        #Load annotations and write a pandas frame
        label = generator.load_annotations(i)
        labeldf = pd.DataFrame(label)
        
        #Add tilename and window ID
        labeldf['tile'] = generator.row["tile"]
        
        labeldf['window'] = i
        
        #Add utm position
        hdf5_file["utm_coords"][i,...] = generator.utm_from_window()
        
        #add to labels
        labels[i] = labeldf
    
    #Write labels to pandas frame
    labeldf = pd.concat(labels, ignore_index=True)
    csv_filename = os.path.join(destination_dir, tilename + ".csv")    
    labeldf.to_csv(csv_filename, index=False)
    
    #Write geographic position 
    #Need to close h5py?
    hdf5_file.close()
    
    #flush system
    sys.stdout.flush()
    
    return "{} completed".format(tilename)



    
    