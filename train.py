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
import models
import keras

##Set seed for reproducibility##
np.random.seed(2)

#Load data
data_paths=glob.glob(config['bbox_data_dir']+"/*.csv")
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
training_generator = DataGenerator(partition['train'], labels, **config['training_params'])
testing_generator = DataGenerator(partition['test'], labels, **config['training_params']) 

#Load Model
DeepForest=models.rgb()

#set loss
DeepForest.compile(loss="binary_crossentropy",optimizer=keras.optimizers.RMSprop(), metrics=['acc'])

#callbacks
callbacks=keras.callbacks.TensorBoard(log_dir='logs')

# Train model on dataset
DeepForest.fit_generator(generator=training_generator,
                    validation_data=testing_generator,
                    use_multiprocessing=True,
                    workers=3,callbacks=callbacks)

#save model
model.save('DeepForest.h5')
