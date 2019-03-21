"""
Train a series of models based on a set of sites. 
For example, retrain on each epoch of pretraining data
"""
from comet_ml import Experiment
import sys
import os
from datetime import datetime
import glob
import pandas as pd 
import copy

#insert path 
from DeepForest.config import load_config
from DeepForest.utils.generators import load_retraining_data
from train import main as training_main
from eval import main as eval_main

#load config - clean
original_DeepForest_config = load_config()       

pretraining_models = {"SJER":"/orange/ewhite/b.weinstein/retinanet/20190318_144257/resnet50_02.h5",
                   "TEAK":"/orange/ewhite/b.weinstein/retinanet/20190315_150652/resnet50_02.h5",
                  "All": "/orange/ewhite/b.weinstein/retinanet/20190314_150323/resnet50_03.h5"}
#pretraining_models = {"SJER" : "/Users/ben/Documents/DeepLidar/snapshots/TEAK_20190125_125012_fullmodel.h5"}

sites = [["SJER"],["TEAK"],["SJER","TEAK"]]

#For each site, match the hand annotations with the pretraining model
results = []
for pretraining_site in pretraining_models:
    
    pretrain_model_path = pretraining_models[pretraining_site]
    for site in sites:
        
        print("Running pretraining site {} with hand annotations {}".format(pretraining_site, site))
        
        #load config - clean
        DeepForest_config = copy.deepcopy(original_DeepForest_config)      
        
        ##Replace config file and experiment
        DeepForest_config["hand_annotation_site"] = site
        DeepForest_config["evaluation_site"] = site
        
        experiment = Experiment(api_key="ypQZhYfs3nSyKzOfz13iuJpj2", project_name='deeplidar', log_code=False)
        experiment.log_parameter("mode","training_grid")   
        
        ###Log experiments
        #experiment.log_parameters(DeepForest_config)    
        dirname = datetime.now().strftime("%Y%m%d_%H%M%S")        
        experiment.log_parameter("Start Time", dirname)    
        
        ##Make a new dir and reformat args
        save_snapshot_path = DeepForest_config["save_snapshot_path"]+ dirname            
        save_image_path = DeepForest_config["save_image_path"]+ dirname
        os.mkdir(save_snapshot_path)        
        
        if not os.path.exists(save_image_path):
            os.mkdir(save_image_path)        
        
        #Load retraining data
        data = load_retraining_data(DeepForest_config)
        print("Before training")
        for x in site:
            DeepForest_config[x]["h5"] = os.path.join(DeepForest_config[x]["h5"],"hand_annotations")
            print(DeepForest_config[x]["h5"])
        
        args = [
            "--epochs", str(DeepForest_config['epochs']),
            "--batch-size", str(DeepForest_config['batch_size']),
            "--backbone", str(DeepForest_config["backbone"]),
            "--score-threshold", str(DeepForest_config["score_threshold"]),
            "--save-path", save_image_path,
            "--snapshot-path", save_snapshot_path,
            "--weights", str(pretrain_model_path)
        ]
    
        #Run training, and pass comet experiment class
        model = training_main(args=args, data=data, DeepForest_config=DeepForest_config, experiment=experiment)  
        
        #Run eval
        experiment = Experiment(api_key="ypQZhYfs3nSyKzOfz13iuJpj2", project_name='deeplidar', log_code=False)
        experiment.log_parameter("mode","evaluation_grid")
            
        args = [
            "--batch-size", str(DeepForest_config['batch_size']),
            '--score-threshold', str(DeepForest_config['score_threshold']),
            '--suppression-threshold', '0.1', 
            '--save-path', 'snapshots/images/', 
            '--model', model, 
            '--convert-model'
        ]
                   
        stem_recall, mAP = eval_main(data = data, DeepForest_config = DeepForest_config, experiment = experiment, args = args)
        results.append({"Evaluation Site" : site, "Pretraining Site": pretraining_site, "Stem Recall": stem_recall, "mAP": mAP})
        
results = pd.DataFrame(results)

results.to_csv("analysis/site_grid" + ".csv")        
        