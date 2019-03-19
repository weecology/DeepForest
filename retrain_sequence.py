"""
Train a series of models based on a sequence of parameters. 
Create a "dilution curve". For example, retrain on each epoch of pretraining data
"""
from comet_ml import Experiment
import sys
import os
from datetime import datetime
import glob
import pandas as pd

#insert path 
from DeepForest.config import load_config
from DeepForest.utils.generators import load_retraining_data
from train import main as training_main
from eval import main as eval_main
from eval import parse_args

#load config
DeepForest_config = load_config()

#find models
#models = glob.glob("/orange/ewhite/b.weinstein/retinanet/20190315_150652/*.h5")
models = glob.glob("/Users/Ben/Documents/DeepLidar/snapshots/*.h5")
    
#For each model, match the hand annotations with the pretraining model
results = []
for model in models:
    
    #Replace config file and experiment
    experiment = Experiment(api_key="ypQZhYfs3nSyKzOfz13iuJpj2", project_name='deeplidar', log_code=False)
    experiment.log_parameter("mode","retrain_sequence")
    
    #Log experiments
    dirname = datetime.now().strftime("%Y%m%d_%H%M%S")  
    experiment.log_parameters(DeepForest_config)
    experiment.log_parameter("Start Time", dirname)    
    
    #Make a new dir and reformat args
    save_snapshot_path = os.path.join(DeepForest_config["save_snapshot_path"], dirname)            
    save_image_path = os.path.join(DeepForest_config["save_image_path"], dirname)           
    os.mkdir(save_snapshot_path)       
    
    if not os.path.exists(save_image_path):
        os.mkdir(save_image_path)        
    
    #Load retraining data
    data = load_retraining_data(DeepForest_config)     
    for site in DeepForest_config["hand_annotation_site"]:
        DeepForest_config[site]["h5"] = os.path.join(DeepForest_config[site]["h5"],"hand_annotations")
        
    args = [
        "--epochs", str(DeepForest_config['epochs']),
        "--batch-size", str(DeepForest_config['batch_size']),
        "--backbone", str(DeepForest_config["backbone"]),
        "--score-threshold", str(DeepForest_config["score_threshold"]),
        "--save-path", save_image_path,
        "--snapshot-path", save_snapshot_path,
        "--weights", str(model)
    ]
    
    #Run training, and pass comet experiment class    
    model = training_main(args, data, DeepForest_config, experiment=experiment)  
    
    #Format output
    experiment = Experiment(api_key="ypQZhYfs3nSyKzOfz13iuJpj2", project_name='deeplidar', log_code=False)      
    experiment.log_parameter("mode","retrain_sequence_evaluation")   
    
    #TODO error here, try below from eval.py
    #pass an args object instead of using command line        
    retinanet_args = [
        "--batch-size", str(DeepForest_config['batch_size']),
        '--score-threshold', str(DeepForest_config['score_threshold']),
        '--suppression-threshold', '0.1', 
        '--save-path', 'snapshots/images/', 
        '--model', model, 
        '--convert-model'
    ]
        
    stem_recall, mAP = eval_main(data, DeepForest_config, experiment, retinanet_args)
    
    model_name = os.path.splitext(os.path.basename(model))[0]    
    results.append({"Model": model_name, "Stem Recall": stem_recall, "mAP": mAP})

results = pd.DataFrame(results)
print(results)
results.to_csv("analysis/pretraining_size" + ".csv")    
    
