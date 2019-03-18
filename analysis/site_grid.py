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
from train import main as training_main
from eval import main as eval_main

#load config
DeepForest_config = load_config("..")

# parse arguments
if args is None:
    args = sys.argv[1:]
args = parse_args(args)

#TODO insert paths here
pretraining_models = {"SJER":"/Users/ben/Documents/DeepLidar/snapshots/SJER_20190301_142501_handannotation.h5",
                      "TEAK":"/Users/ben/Documents/DeepLidar/snapshots/SJER_20190301_142501_handannotation.h5"}
sites = ["TEAK","SJER"]
#For each site, match the hand annotations with the pretraining model
for pretraining_site in pretraining_models:
    
    model = pretraining_models[pretraining_site]
    for site in sites:
        
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
        model = training_main(args, data, DeepForest_config, experiment=experiment)  
        
        #Run evaluation
        #Run eval
        stem_recall, mAP = eval_main(DeepForest_config, args)
        results.append({"Evaluation Site" : site, "Pretraining Site": pretraining_site, "Stem Recall": stem_recall, "mAP": mAP})
        
        results = pd.DataFrame(results)
        
        #model name
        model_name = os.path.splitext(os.path.basename(mode.saved_model))[0]
        results.to_csv("site_grid" + ".csv")        
        