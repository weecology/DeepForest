import yaml

with open('_config.yml', 'r') as f:
    config = yaml.load(f)
    if config is None:
        print("No config file found")