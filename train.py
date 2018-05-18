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
from models import rgb
import keras

##Set seed for reproducibility##
np.random.seed(2)

#Load data
data_paths=glob.glob(config['bbox_data_dir']+"/*.csv")
dataframes = (pd.read_csv(f,index_col=0) for f in data_paths)
data = pd.concat(dataframes, ignore_index=True)

#one data check #TODO understand why this happens, rare boxes of 0 size?
data=data[data.xmin!=data.xmax]

#set index explicitely
data=data.set_index('box')

#create numeric labels
lookup={"Background": 0, "Tree": 1}
data['label_numeric']=[lookup[x] for x in data.label]

#Partition data
msk = np.random.rand(len(data)) < 0.8

#Training and testing dataframes
train = data[msk]
test = data[~msk]

#Create dictionaries to keep track of labels and splits
partition={"train": train.index.values,"test": test.index.values}
labels=data.label_numeric.to_dict()

# Generators
training_generator = DataGenerator(box_file=train, list_IDs=partition['train'], labels=labels, **config['training_params'])
testing_generator =DataGenerator(box_file=test, list_IDs=partition['test'], labels=labels, **config['training_params'])

#Load Model
DeepForest=rgb.get_model(is_training=True)

#set loss
DeepForest.compile(loss="binary_crossentropy",optimizer=keras.optimizers.RMSprop(), metrics=['acc'])

#callbacks
callbacks=keras.callbacks.TensorBoard(log_dir='logs',write_images=True)

# Train model on dataset
DeepForest.fit_generator(generator=training_generator,
                    validation_data=testing_generator, epochs=200,
                    use_multiprocessing=False,
                    workers=1,callbacks=[callbacks])
