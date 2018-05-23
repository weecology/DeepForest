'''
Training script for DeepForest.
Ben Weinstein - ben.weinstein@weecology.org
Load data, partition into training and testing, and evaluate deep learning model
'''
from comet_ml import Experiment
from DeepForest.config import config
import pandas as pd
import glob
import numpy as np
from DeepForest.CropGenerator import DataGenerator
from models import rgb
import keras
from datetime import datetime
from DeepForest.tools import TimeHistory, PlotImages
import tensorflow as tf

#set experiment and log configs
experiment = Experiment(api_key="ypQZhYfs3nSyKzOfz13iuJpj2",project_name='deepforest')
experiment.log_multiple_params(config['data_generator_params'])
experiment.log_multiple_params(config['training'])

##Set seed for reproducibility##
np.random.seed(2)

#Load data and comine into a large 
data_paths=glob.glob(config['bbox_data_dir']+"/*.csv")
dataframes = (pd.read_csv(f,index_col=0) for f in data_paths)
data = pd.concat(dataframes, ignore_index=True)

#set index explicitely
data=data.set_index('box')

#create numeric labels
lookup={"Background": 0, "Tree": 1}
data['label_numeric']=[lookup[x] for x in data.label]

#optionally subset data, if config argument is numeric, subset data
if(not isinstance(config["subsample"],str)):
    data=data.sample(n=config["subsample"], random_state=2)
    
#Partition data
msk = np.random.rand(len(data)) < 0.8

#Training and testing dataframes
train = data[msk]
test = data[~msk]

#log data size
experiment.log_parameter("training_samples", train.shape[0])
experiment.log_parameter("testing_samples", test.shape[0])

#Create dictionaries to keep track of labels and splits
partition={"train": train.index.values,"test": test.index.values}
labels=data.label_numeric.to_dict()

# Generators
training_generator = DataGenerator(box_file=train, list_IDs=partition['train'], labels=labels, **config['data_generator_params'])
testing_generator =DataGenerator(box_file=test, list_IDs=partition['test'], labels=labels, **config['data_generator_params'])

#Load Model
DeepForest=rgb.get_model(is_training=True)

#set loss
DeepForest.compile(loss="binary_crossentropy",optimizer=keras.optimizers.Adam(), metrics=['acc'])

#Callbacks and objects to add to comet logging
time_callback = TimeHistory(experiment)

#now=datetime.now()
#keras.callbacks.TensorBoard(log_dir='logs/'+ now.strftime("%Y%m%d-%H%M%S") + '/',write_images=True)

# Train model on dataset
#samples/batchsize
steps_per_epoch=int(train.shape[0]/config['data_generator_params']['batch_size'])

DeepForest.fit_generator(generator=training_generator,
                         validation_data=testing_generator,
                         workers=config['training']['workers'],
                         epochs=config['training']['epochs'],
                         use_multiprocessing=True,
                         steps_per_epoch=steps_per_epoch,
                         validation_steps=steps_per_epoch,
                         callbacks=[time_callback])
