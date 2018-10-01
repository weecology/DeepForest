'''
Preprocess data
Loading, Standardizing and Filtering the raw data to dimish false positive labels 
'''

import pandas as pd
import glob
import os
import random
import xmltodict
import numpy as np
import rasterio
from PIL import Image
import slidingwindow as sw
import itertools

def load_data(data_dir,res):
    '''
    data_dir: path to .csv files. Optionall can be a path to a specific .csv file.
    res: Cell resolution of the rgb imagery
    '''
    
    if(os.path.splitext(data_dir)[-1]==".csv"):
        data=pd.read_csv(data_dir,index_col=0)
    else:
        #Gather list of csvs
        data_paths=glob.glob(data_dir+"/*.csv")
        dataframes = (pd.read_csv(f,index_col=0) for f in data_paths)
        data = pd.concat(dataframes, ignore_index=False)
    
    #Modify indices, which came from R, zero indexed in python
    data=data.set_index(data.index.values-1)
    data.numeric_label=data.numeric_label-1
    
    #Remove xmin==xmax
    data=data[data.xmin!=data.xmax]    
    data=data[data.ymin!=data.ymax]    

    ##Create bounding coordinates with respect to the crop for each box
    #Rescaled to resolution of the cells.Also note that python and R have inverse coordinate Y axis, flipped rotation.
    data['origin_xmin']=(data['xmin']-data['tile_xmin'])/res
    data['origin_xmax']=(data['xmin']-data['tile_xmin']+ data['xmax']-data['xmin'])/res
    data['origin_ymin']=(data['tile_ymax']-data['ymax'])/res
    data['origin_ymax']= (data['tile_ymax']-data['ymax']+ data['ymax'] - data['ymin'])/res  
        
    return(data)
    
def zero_area(data):
    data=data[data.xmin!=data.xmax]    
    return(data)

#Filter by ndvi threshold
def NDVI(data,threshold,data_dir):
    
    #for each row
    for row,index in data.iterrows():
        
        #create the hyperspectral object
        h=Hyperspectral(data_dir + row['hyperspec_path'])
        
        #create clipExtent from box
        clipExtent={}
        clipExtent["xmin"]=row["xmin"]
        clipExtent["ymin"]=row["ymin"]
        clipExtent["xmax"]=row["xmax"]
        clipExtent["ymax"]=row["ymax"]
        
        #Calculate NDVI
        NDVI=f.NDVI(clipExtent=clipExtent)
        
        data['NDVI']=NDVI
    
    #Create lower bound for NDVI   
    data=data[data.NDVI > threshold]    
    return(data)

def load_xml(path,res):

    #parse
    with open(path) as fd:
        doc = xmltodict.parse(fd.read())
    
    #grab objects
    tile_xml=doc["annotation"]["object"]
    
    xmin=[]
    xmax=[]
    ymin=[]
    ymax=[]
    label=[]
    
    if type(tile_xml) == list:
        
        treeID=np.arange(len(tile_xml))
        
        #Construct frame if multiple trees
        for tree in tile_xml:
            xmin.append(tree["bndbox"]["xmin"])
            xmax.append(tree["bndbox"]["xmax"])
            ymin.append(tree["bndbox"]["ymin"])
            ymax.append(tree["bndbox"]["ymax"])
            label.append(tree['name'])
    else:
        
        #One tree
        treeID=0
        
        xmin.append(tile_xml["bndbox"]["xmin"])
        xmax.append(tile_xml["bndbox"]["xmax"])
        ymin.append(tile_xml["bndbox"]["ymin"])
        ymax.append(tile_xml["bndbox"]["ymax"])
        label.append(tile_xml['name'])        
        
    rgb_path=doc["annotation"]["filename"]
    
    #bounds
    
    #read in tile to get dimensions
    full_path=os.path.join("data",doc["annotation"]["folder"] ,rgb_path)

    with rasterio.open(full_path) as dataset:
        bounds=dataset.bounds         
    
    frame=pd.DataFrame({"treeID":treeID,"xmin":xmin,"xmax":xmax,"ymin":ymin,"ymax":ymax,"rgb_path":rgb_path,"label":label,
                        "numeric_label":0,
                        "tile_xmin":bounds.left,
                        "tile_xmax":bounds.right,
                        "tile_ymin":bounds.bottom,
                        "tile_ymax":bounds.top})

    #Modify indices, which came from R, zero indexed in python
    frame=frame.set_index(frame.index.values)

    ##Match expectations of naming, no computation needed for hand annotations
    frame['origin_xmin']=frame["xmin"].astype(float)
    frame['origin_xmax']=frame["xmax"].astype(float)
    frame['origin_ymin']=frame["ymin"].astype(float)
    frame['origin_ymax']= frame["ymax"].astype(float)
    
    return(frame)

def compute_windows(image,pixels=250,overlap=0.05):
    im = Image.open(image)
    numpy_image = np.array(im)    
    windows = sw.generate(numpy_image, sw.DimOrder.HeightWidthChannel, pixels,overlap )
    return(windows)

def expand_grid(data_dict):
    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())

def split_training(data,DeepForest_config,experiment,single_tile=False):
    
    '''
    Divide windows into training and testing split. Assumes that all tiles have the same size.
    '''
    
    #Compute list of sliding windows, assumed that all objects are the same extent and resolution
    base_dir=DeepForest_config["evaluation_tile_dir"]
    image_path=os.path.join(base_dir, data.rgb_path.unique()[0])
    windows=compute_windows(image=image_path, pixels=DeepForest_config["patch_size"], overlap=DeepForest_config["patch_overlap"])
    
    #Compute Windows
    #Create dictionary of windows for each image
    tile_windows={}
    
    all_images=list(data.rgb_path.unique())

    tile_windows["image"]=all_images
    tile_windows["windows"]=np.arange(0,len(windows))
    
    #Expand grid
    tile_data=expand_grid(tile_windows)

    #For retraining there is only a single tile
    if single_tile:
        
        #Shuffle data if needed
        if DeepForest_config["shuffle_training"]:
            tile_data.sample(frac=1)
        else:
            np.random.seed(2)
            
        #Split training and testing
        msk = np.random.rand(len(tile_data)) < 1-(float(DeepForest_config["validation_percent"])/100)
        training = tile_data[msk]
        evaluation=tile_data[~msk]    
    else:
        
        #select a validation tile, record in log.        
        if DeepForest_config["shuffle_eval"]:
            eval_tile=random.sample(all_images,1)[0]
        else:
            eval_tile=all_images[1]
        
        #Log if not debugging.
        if not experiment==None:
            experiment.log_parameter(eval_tile,"Evaluation Tile")
        
        #Split data based on samples
        evaluation=tile_data[tile_data.image==eval_tile]
        training=tile_data[~(tile_data.image==eval_tile)]
        
        if not DeepForest_config["training_images"]=="All":
            
            #Shuffle if desired - preserve groups of images.
            if DeepForest_config["shuffle_training"]:
                groups = [df for _, df in training.groupby('image')]
                groups=[x.sample(frac=1) for x in groups]      
                random.shuffle(groups)
                training=pd.concat(groups).reset_index(drop=True)
                
            #Select first n windows, reorder to preserve tile order.
            training=training.head(n=DeepForest_config["training_images"])
        
        if not DeepForest_config["evaluation_images"]=="All":
            
            #Shuffle if desired
            if DeepForest_config["shuffle_eval"]:
                evaluation.sample(frac=1)
                
            #Select first n windows, reorder to preserve tile order.
            evaluation=evaluation.head(n=DeepForest_config["evaluation_images"])
            groups = [df for _, df in evaluation.groupby('image')]
            groups=[x.sample(frac=1) for x in groups]
            evaluation=pd.concat(groups).reset_index(drop=True)        
    
    return([training.to_dict("index"),evaluation.to_dict("index")])
    
def NEON_annotations(site,DeepForest_config):
   
    glob_path=os.path.join("data",site,"annotations") + "/" + site + "*.xml"
    xmls=glob.glob(glob_path)
    
    annotations=[]
    for xml in xmls:
        r=load_xml(xml,DeepForest_config["rgb_res"])
        annotations.append(r)

    data=pd.concat(annotations)
    
    #Compute list of sliding windows, assumed that all objects are the same extent and resolution
    image_path=os.path.join("data",site, data.rgb_path.unique()[0])
    windows=compute_windows(image=image_path, pixels=DeepForest_config["patch_size"], overlap=DeepForest_config["patch_overlap"])
    
    #Compute Windows
    #Create dictionary of windows for each image
    tile_windows={}
    
    all_images=list(data.rgb_path.unique())

    tile_windows["image"]=all_images
    tile_windows["windows"]=np.arange(0,len(windows))
    
    #Expand grid
    tile_data=expand_grid(tile_windows)    
    
    return [data,tile_data.to_dict("index")]