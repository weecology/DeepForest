import os
from DeepForest.onthefly_generator import OnTheFlyGenerator
from DeepForest.preprocess import NEON_annotations

def create_NEON_generator(batch_size, DeepForest_config, name="evaluation"):
    """ Create generators for training and validation.
    """
    annotations, windows = NEON_annotations(DeepForest_config)

    #Training Generator
    generator =  OnTheFlyGenerator(
        annotations,
        windows,
        batch_size = batch_size,
        DeepForest_config = DeepForest_config,
        group_method="none",
        name=name)
    
    return(generator)