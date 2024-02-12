# Train
from deepforest import main
import argparse
import yaml
import os
import json
from pytorch_lightning.loggers import CometLogger
import datetime

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
    

def wrapper(config):
    if config["pretrain"]["model"] is None:
        if config["pretrain"]["use_pretrain"]:
            pretrain(train=config["pretrain"]["train"], test=config["pretrain"]["test"])

    m = train(config["train"], config["test"], config["savedir"])

    evaluate(m, config["test"])

def pretrain(train, test):
    pass

def evaluate(m, test):
    m.trainer.validate(m)

def train(train, test, savedir):
    comet_logger = CometLogger(api_key="ypQZhYfs3nSyKzOfz13iuJpj2",
                              project_name="deepforest-pytorch", workspace="bw4sz")
    comet_logger.experiment.add_tag("Training")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comet_logger.experiment.log_parameter("timestamp", timestamp)
    
    try:
        os.mkdir(savedir)
    except FileExistsError:
        pass
    
    m = main.deepforest()
    m.create_trainer(logger=comet_logger)
    comet_logger.experiment.log_parameters(m.config)
    m.trainer.fit(m)
    model_path = "{}/{}.pl".format(savedir, timestamp)
    m.save_model(model_path)
    comet_logger.experiment.log_parameter("model path", model_path)

    return m

if __name__ == "__main__":
    config = read_config("backbone_config.yml")
    wrapper(config)