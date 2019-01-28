import cv2
import os
from DeepForest import config
from DeepForest import onthefly_generator
from DeepForest import preprocess
from DeepForest import Lidar
from DeepForest import utils
import pyfor
import warnings
import sys

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
def sample(n=50):
    """
    Grab n random images from across the site
    """
    #Load config
    DeepForest_config = config.load_config()
    
    #Read in data
    data = preprocess.load_data(data_dir=DeepForest_config['training_csvs'], res=0.1, lidar_path=DeepForest_config["lidar_path"])
    
    #Create windows
    windows = preprocess.create_windows(data, DeepForest_config,base_dir = DeepForest_config["evaluation_tile_dir"])
    
    selected_windows = windows[["tile","window"]].drop_duplicates().sample(n=n)
        
    generator = onthefly_generator.OnTheFlyGenerator(data=data, windowdf=selected_windows, DeepForest_config = DeepForest_config)
    
    folder_dir = os.path.join("data",DeepForest_config["evaluation_site"],"samples")
    
    if not os.path.exists(folder_dir):
        os.mkdir(folder_dir)
                 
    for i in range(generator.size()):
        
        #Load image - done for side effects
        four_channel = generator.load_image(i)
        
        if generator.clipped_las == None:
            continue
            
        #name RGB
        tilename = os.path.splitext(generator.image_data[i]["tile"])[0]
        tilename = tilename + "_" + str(generator.image_data[i]["window"]) + ".tif"
        filename = os.path.join(folder_dir, tilename)
        
        #Write
        cv2.imwrite(filename, generator.image)
                
        #name .laz
        tilename = os.path.splitext(generator.image_data[i]["tile"])[0]
        tilename = tilename + "_" + str(generator.image_data[i]["window"]) + ".laz"
        filename = os.path.join(folder_dir, tilename)        
        
        
        #Write .laz
        generator.clipped_las.write(filename)
        
if __name__ == "__main__":
    sample(n=50)