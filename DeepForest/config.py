import yaml
import os

def load_config():
      #seperate config loads for local, HPC
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
        return(config)