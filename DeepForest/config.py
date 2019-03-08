import yaml
import os

def load_config(dir=None):

        #check location, if running locally, use tiny test file for debugging
        if os.uname().nodename == "MacBook-Pro.local":
                yaml_name = "_config_debug.yml"
        else:
                yaml_name = "_config.yml"
                
        if dir:
                yaml_name = os.path.join(dir, yaml_name)

        with open(yaml_name, 'r') as f:
                config = yaml.load(f)

        if config is None:
                raise IOError("No config file found")          
        return(config)