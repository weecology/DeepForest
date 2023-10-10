# Train
from deepforest import main
from src import utilities

def wrapper():
    config = utilities.read_config()

    if config["pretrain_model"] is None:
        pretrain()

    train()

def pretrain():
    pass

def train():
    pass
