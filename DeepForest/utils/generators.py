import os
import glob
import pandas as pd
from DeepForest.onthefly_generator import OnTheFlyGenerator
from DeepForest.preprocess import NEON_annotations, load_csvs
from DeepForest import Generate

def load_retraining_data(DeepForest_config):
    """
    Overall function to find training data based on config file
    mode: retrain
    """    
    #for each hand_annotation tile, check if its been generated.
    for site in DeepForest_config["hand_annotation_site"]:
        RGB_dir = DeepForest_config[site]["hand_annotations"]["RGB"]
        h5_dirname = os.path.join(DeepForest_config[site]["h5"], "hand_annotations")
        
        #Check if hand annotations have been generated. If not create H5 files.
        path_to_handannotations = []
        if os.path.isdir(RGB_dir):
            tilenames = glob.glob(os.path.join(RGB_dir,"*.tif"))
        else:
            tilenames = [os.path.splitext(os.path.basename(RGB_dir))[0]]
            
        for x in tilenames:
            tilename = os.path.splitext(os.path.basename(x))[0]                
            tilename = os.path.join(h5_dirname, tilename) + ".csv"
            path_to_handannotations.append(os.path.join(RGB_dir, tilename))            
                
        #for each annotation, check if exists in h5 dir
        for index, path in enumerate(path_to_handannotations):
            if not os.path.exists(path):
                
                #Generate xml name, assumes annotations are one dir up from rgb dir
                annotation_dir = os.path.join(os.path.dirname(os.path.dirname(RGB_dir)),"annotations")
                annotation_xmls = os.path.splitext(os.path.basename(tilenames[index]))[0] + ".xml"
                full_xml_path = os.path.join(annotation_dir, annotation_xmls )
                
                print("Generating h5 for hand annotated data from tile {}".format(tilename))                
                Generate.run(tile_xml = full_xml_path, mode="retrain", site = site)
        
    #combine data across sites        
    dataframes = []
    for site in DeepForest_config["hand_annotation_site"]:
        h5_dirname = os.path.join(DeepForest_config[site]["h5"], "hand_annotations")
        df = load_csvs(h5_dirname)
        df["site"] = site
        dataframes.append(df)
    data = pd.concat(dataframes, ignore_index=True)      
    
    return data

def load_training_data(DeepForest_config):
    """
    Overall function to find training data based on config file
    mode: train
    """
    #For each training directory (optionally more than one site)
    dataframes = []
    for site in DeepForest_config["pretraining_site"]:
        h5_dirname = DeepForest_config[site]["h5"]
        df = load_csvs(h5_dirname)
        df["site"] = site
        dataframes.append(df)
        
    #Create a dict assigning the tiles to the h5 dir
    data = pd.concat(dataframes, ignore_index=True)         
    return data

def create_NEON_generator(batch_size, DeepForest_config, name="evaluation"):
    """ Create generators for training and validation.
    """
    annotations, windows = NEON_annotations(DeepForest_config)

    #Training Generator
    generator =  OnTheFlyGenerator(
        annotations,
        windows,
        batch_size = batch_size,
        DeepForest_config = DeepForest_config,
        group_method="none",
        name=name)
    
    return(generator)