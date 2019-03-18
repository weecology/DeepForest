"""
Train a series of models based on a set of sites. 
For example, retrain on each epoch of pretraining data
"""
from comet_ml import Experiment
import sys
sys.path.append("..")
import os
from datetime import datetime
import glob

#insert path 
from DeepForest.config import load_config
from DeepForest.utils.generators import load_retraining_data
from train import main

#load config
DeepForest_config = load_config("..")

#For each site, match the hand annotations with the pretraining model
for site in ["TEAK","SJER"]:
    
    #Replace config file and experiment
    DeepForest_config["hand_annotation_site"] = [site]
    DeepForest_config["evaluation_site"] = [site]
    experiment = Experiment(api_key="ypQZhYfs3nSyKzOfz13iuJpj2", project_name='deeplidar', log_code=False)
    
    #Log experiments
    experiment.log_parameters(DeepForest_config)    
    experiment.log_parameter("Start Time", dirname)    
    experiment.log_parameter("Training Mode", mode.mode)
    
    #Make a new dir and reformat args
    dirname = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_snapshot_path=DeepForest_config["save_snapshot_path"]+ dirname            
    save_image_path =DeepForest_config["save_image_path"]+ dirname
    os.mkdir(save_snapshot_path)        
    if not os.path.exists(save_image_path):
        os.mkdir(save_image_path)        
    
    #Load retraining data
    data = load_retraining_data(DeepForest_config)     
    for site in DeepForest_config["hand_annotation_site"]:
        DeepForest_config[site]["h5"] = os.path.join(DeepForest_config[site]["h5"],"hand_annotations")
    
    args = [
        "--epochs", DeepForest_config['epochs'],
        "--batch-size", str(DeepForest_config['batch_size']),
        "--backbone", str(DeepForest_config["backbone"]),
        "--score-threshold", str(DeepForest_config["score_threshold"]),
        "--save-path", save_image_path,
        "--snapshot-path", save_snapshot_path,
        "--weights", str(DeepForest_config["weights"])
    ]

    #Run training, and pass comet experiment class
    main(args, data, DeepForest_config, experiment=experiment)  