#Deepforest Preprocessing model
'''
The preprocessing module is used to reshape data into format suitable for training or prediction. For example cutting large tiles into smaller images.
'''
import pandas as pd
from PIL import Image
import numpy as np
import os
import slidingwindow

def image_name_from_path(image_path):
    """
    Convert path to image name for use in indexing
    """
    image_name = os.path.basename(image_path)
    image_name = os.path.splitext(image_name)[0]
    
    return image_name

def compute_windows(numpy_image, patch_size, patch_overlap):
    ''''
    Create a sliding window object from a raster tile
    Args:
        numpy_image (array): Raster object as numpy array to cut into crops
    Returns:
        windows (list): a sliding windows object
    '''
    
    #Generate overlapping sliding windows
    windows = slidingwindow.generate(numpy_image, slidingwindow.DimOrder.HeightWidthChannel, patch_size, patch_overlap)
    
    return(windows)

def select_annotations(annotations, windows, index):
    """
    Select annotations that overlap with selected image crop
    Args:
        image_name (str): Name of the image in the annotations file to lookup. 
        annotations_file: path to annotations file in the format -> image_path, xmin, ymin, xmax, ymax, label
        windows: A sliding window object (see compute_windows) 
        index: The index in the windows object to use a crop bounds
    """
    
    # Window coordinates - with respect to tile
    window_xmin, window_ymin, w, h= windows[index].getRect()
    window_xmax = window_xmin + w
    window_ymax = window_ymin + h
    
    #buffer coordinates a bit to grab boxes that might start just against the image edge. Don't allow boxes that start and end after the offset
    offset = 5
    selected_annotations = annotations[ 
        (annotations.xmin > (window_xmin -offset)) &  
        (annotations.xmin < (window_xmax)) &
        (annotations.xmax > (window_xmin)) &                        
        (annotations.ymin > (window_ymin - offset))  &
        (annotations.xmax < (window_xmax + offset)) &
        (annotations.ymin < (window_ymax)) &  
        (annotations.ymax > (window_ymin)) &                
        (annotations.ymax < (window_ymax + offset))].copy()
    
    #change the image name
    image_name = os.path.splitext("{}".format(annotations.image_path.unique()[0]))[0]
    image_basename = os.path.splitext(image_name)[0]
    selected_annotations.image_path = "{}_{}.jpg".format(image_basename,index) 
    
    #update coordinates with respect to origin
    selected_annotations.xmax = (selected_annotations.xmin - window_xmin)  + (selected_annotations.xmax - selected_annotations.xmin)    
    selected_annotations.xmin = (selected_annotations.xmin - window_xmin)
    selected_annotations.ymax = (selected_annotations.ymin - window_ymin)  + (selected_annotations.ymax - selected_annotations.ymin)    
    selected_annotations.ymin = (selected_annotations.ymin - window_ymin)
    
    #cut off any annotations over the border.
    selected_annotations.xmin[selected_annotations.xmin < 0] = 0
    selected_annotations.xmax[selected_annotations.xmax > w] = w   
    selected_annotations.ymin[selected_annotations.ymin < 0] = 0
    selected_annotations.ymax[selected_annotations.ymax > h] = h   
    
    return selected_annotations
 
def save_crop(base_dir, image_name, index, crop):
    """
    Save window crop as image file to be read by PIL. Filename should match the image_name + window index
    """
    #create dir if needed
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    im = Image.fromarray(crop)
    image_basename = os.path.splitext(image_name)[0]    
    filename = "{}/{}_{}.jpg".format(base_dir, image_basename, index)
    im.save(filename)   

def split_training_raster(path_to_raster, annotations_file, base_dir, patch_size, patch_overlap):
    """
    Divide a large tile into smaller arrays for training. Each crop will be saved to file
    Args:
        path_to_tile (str): Path to a tile that can be read by rasterio on disk
        annotations_file (str): Path to annotations file with data in the format -> image_path, xmin, ymin, xmax, ymax, label
        base_dir (str): Where to save the annotations and image crops relative to current working dir
        patch_size (int): Maximum dimensions of square window
        patch_overlap (float): Percent of overlap among windows 0->1
    Returns:
        A pandas dataframe with annotations file for training.
    """
    #try and raise path to raster is fail
    #Load raster as image
    raster = Image.open(path_to_raster)
    numpy_image = np.array(raster)        
    
    #Compute sliding window index
    windows = compute_windows(numpy_image, patch_size, patch_overlap)
    
    #Get image name for indexing
    image_name = os.path.basename(path_to_raster)
    
    # Load annotations file
    annotations = pd.read_csv(annotations_file)
    
    #open annotations file
    image_annotations = annotations[annotations.image_path == image_name].copy()
    
    if not annotations.shape[1]==6:
        raise ValueError("Annotations file has {} columns, should have format image_path, xmin, ymin, xmax, ymax, label".format(annotations.shape[1]))
            
    annotations_files = []
    for index, window in enumerate(windows):
        
        #Crop image
        crop = numpy_image[windows[index].indices()]
        
        #Find annotations, image_name is the basename of the path
        crop_annotations = select_annotations(image_annotations, windows, index)
        
        #save annotations
        annotations_files.append(crop_annotations)
        
        #save image crop
        save_crop(base_dir, image_name, index, crop)
        
    annotations_files = pd.concat(annotations_files)
    
    #Checkpoint csv files, useful for parallelization    
    #Use filename of the raster path to save the annotations
    image_basename = os.path.splitext(image_name)[0]    
    file_path = image_basename + ".csv"
    file_path = os.path.join(base_dir, file_path)
    annotations_files.to_csv(file_path, index=False, header=False)
    
    return annotations_files