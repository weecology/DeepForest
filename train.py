'''
Training script for DeepForest.
Ben Weinstein - ben.weinstein@weecology.org
Load data, partition into training and testing, and evaluate deep learning model
'''
from DeepForest.config import config
import pandas as pd
import glob
import numpy as np
from DeepForest.CropGenerator import DataGenerator
import keras

##Set seed for reproducibility##
np.random.seed(2)

#Load data
data_paths=glob.glob(config['data_dir']+"/*.csv")
dataframes = (pd.read_csv(f,index_col=0) for f in data_paths)
data = pd.concat(dataframes, ignore_index=True)

#set index explicitely
data=data.set_index('box')

#Partition data
msk = np.random.rand(len(data)) < 0.8

#Training and testing dataframes
train = data[msk]
test = data[~msk]

#Create dictionaries to keep track of labels and splits
partition={"train": train.index.values,"test": test.index.values}
labels=data.label.to_dict()

# Generators
training_generator = DataGenerator(partition['train'], labels, **config.training_params)
testing_generator = DataGenerator(partition['test'], labels, **config.training_params) 

#Load Model
#DeepForest...

# Train model on dataset
DeepForest.fit_generator(generator=training_generator,
                    validation_data=testing_generator,
                    use_multiprocessing=True,
                    workers=6)


