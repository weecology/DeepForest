import yaml
import os

def load_config(name):
      #seperate config loads for local, HPC, for training and retraining.
        if name=="train":
                 
                #check location, if running locally, use tiny test file for debugging
                if os.uname().nodename == "MacBook-Pro.local":
                        with open('_config_debug.yml', 'r') as f:
                                config = yaml.load(f)
                                if config is None:
                                        print("No config file found")
                else:
                        with open('_config.yml', 'r') as f:
                                config = yaml.load(f)
                                if config is None:
                                        print("No config file found")
        if name=="retrain":
                #check location, if running locally, use tiny test file for debugging
                if os.uname().nodename == "MacBook-Pro.local":
                        with open('_retrain_config_debug.yml', 'r') as f:
                                config = yaml.load(f)
                                if config is None:
                                        print("No config file found")
                else:
                        with open('_retrain_config.yml', 'r') as f:
                                config = yaml.load(f)
                                if config is None:
                                        print("No config file found")                
        return(config)