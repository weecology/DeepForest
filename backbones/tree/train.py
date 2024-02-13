# Train
from deepforest import main
import argparse
import yaml
import os
import json
from pytorch_lightning.loggers import CometLogger
from datetime import datetime
import torch

def read_config(config_path):
    """Read config yaml file
    Args:
        config_path
        config
    """
    #Allow command line to override 
    parser = argparse.ArgumentParser("DeepForest Tree Training config")
    parser.add_argument('-d', '--my-dict', type=json.loads, default=None)
    args = parser.parse_known_args()
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

    except Exception as e:
        raise FileNotFoundError("There is no config at {}, yields {}".format(
            config_path, e))
    
    #Update anything in argparse to have higher priority
    if args[0].my_dict:
        for key, value in args[0].my_dict:
            config[key] = value
        
    return config
    

def wrapper(config, deepforest_config_kwargs=None, pretrain_kwargs=None):
    if config["pretrain"]["model"] is None:
        if config["pretrain"]["use_pretrain"]:
            m = pretrain(
                train=config["pretrain"]["train"],
                test=config["pretrain"]["test"],
                pretrain_kwargs=pretrain_kwargs,
                sample_limit=config["pretrain"["sample_limit"]],
                groupby=config["pretrain"]["groupby"])
            checkpoint = m.model_path
        else:
            checkpoint = None

    m = train(config["train"], config["test"], config["savedir"], deepforest_config_kwargs, checkpoint=checkpoint)

    evaluate(m, config["test"])

    return m

def pretrain(train, test, savedir, groupby=None, sample_limit=None, model_path=None, pretrain_kwargs=None):
    #comet_logger = CometLogger(api_key="ypQZhYfs3nSyKzOfz13iuJpj2",
    #                          project_name="deepforest-pytorch", workspace="bw4sz")
    #comet_logger.experiment.add_tag("Pretraining")
    comet_logger = None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #comet_logger.experiment.log_parameter("timestamp", timestamp)
    
    try:
        os.mkdir(savedir)
    except FileExistsError:
        pass
    
    if type(train) == str:
        m = main.deepforest(config_args=pretrain_kwargs)
        m.config["train"]["csv_file"] = train
        m.config["train"]["root_dir"] = os.path.dirname(train)
        m.config["validation"]["csv_file"] = test
        m.config["validation"]["root_dir"] = os.path.dirname(test)
    elif type(train) == torch.utils.data.DataLoader:
        m = main.deepforest(existing_train_dataloader=train, existing_val_dataloader=test)

    #hard code while in the plane
    m.config["train"]["fast_dev_run"] = True
    m.create_trainer(logger=comet_logger)
    #comet_logger.experiment.log_parameters(m.config)
    m.trainer.fit(m)
    model_path = "{}/{}.pl".format(savedir, timestamp)
    m.model_path = model_path
    m.save_model(model_path)
    #comet_logger.experiment.log_parameter("model path", model_path)
    
    return m

def evaluate(m, test):
    m.trainer.validate(m)

def train(train, test, savedir, deepforest_config_kwargs, checkpoint=None):
    #comet_logger = CometLogger(api_key="ypQZhYfs3nSyKzOfz13iuJpj2",
    #                          project_name="deepforest-pytorch", workspace="bw4sz")
    #comet_logger.experiment.add_tag("Training")
    comet_logger = None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #comet_logger.experiment.log_parameter("timestamp", timestamp)
    
    try:
        os.mkdir(savedir)
    except FileExistsError:
        pass
    
    if checkpoint:
        main.deepforest.load_from_checkpoint(checkpoint)

    if type(train) == str:
        m = main.deepforest(config_args=deepforest_config_kwargs)
        m.config["train"]["csv_file"] = train
        m.config["train"]["root_dir"] = os.path.dirname(train)
        m.config["validation"]["csv_file"] = test
        m.config["validation"]["root_dir"] = os.path.dirname(test)
    elif type(train) == torch.utils.data.DataLoader:
        m = main.deepforest(existing_train_dataloader=train, existing_val_dataloader=test)

    #hard code while in the plane
    m.config["train"]["fast_dev_run"] = True
    m.create_trainer(logger=comet_logger)
    #comet_logger.experiment.log_parameters(m.config)
    m.trainer.fit(m)
    model_path = "{}/{}.pl".format(savedir, timestamp)
    m.model_path = model_path
    m.save_model(model_path)
    #comet_logger.experiment.log_parameter("model path", model_path)

    return m

if __name__ == "__main__":
    config = read_config("backbone_config.yml")
    wrapper(config)