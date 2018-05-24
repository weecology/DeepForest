'''
Training script for DeepForest.
Ben Weinstein - ben.weinstein@weecology.org
Load data, partition into training and testing, and evaluate deep learning model
'''
from comet_ml import Experiment

import os
import pandas as pd
import glob
import numpy as np
import keras
from datetime import datetime
from DeepForest.config import config
from DeepForest.tools import TimeHistory
from DeepForest.CropGenerator import DataGenerator
from DeepForest import preprocess, evaluate
from models import inception

#set experiment and log configs
experiment = Experiment(api_key="ypQZhYfs3nSyKzOfz13iuJpj2",project_name='deepforest')
experiment.log_multiple_params(config['data_generator_params'])
experiment.log_multiple_params(config['training'])

##Set seed for reproducibility##
np.random.seed(2)

#Load data and comine into a large 
data=preprocess.load_data(data_dir=config['bbox_data_dir'],nsamples=config["subsample"])

##Preprocess Filters##
data=preprocess.zero_area(data)
    
#Partition data in training and testing dataframes
msk = np.random.rand(len(data)) < 0.8
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

######################################
#Callbacks and objects to add to comet logging#
######################################

time_callback = TimeHistory(experiment,nsamples=train.shape[0])

#Create logdir
now=datetime.now()
logdir='logs/'+ now.strftime("%Y%m%d-%H%M%S")
#keras.callbacks.TensorBoard(log_dir=logdir + '/',write_images=True)

###
#Fit
###

#samples/batchsize
steps_per_epoch=int(train.shape[0]/config['data_generator_params']['batch_size'])

#Load Model
DeepForest=inception.get_model()

#set loss
DeepForest.compile(loss="binary_crossentropy",optimizer=keras.optimizers.Adam(), metrics=['acc'])

DeepForest.fit_generator(generator=training_generator,
                         validation_data=testing_generator,
                         workers=config['training']['workers'],
                         epochs=config['training']['epochs'],
                         use_multiprocessing=True,
                         steps_per_epoch=steps_per_epoch,
                         validation_steps=steps_per_epoch,
                         callbacks=[time_callback])

#Write model to file
# serialize model to JSON
if not os.path.exists(logdir):
    os.makedirs(logdir)
    
model_json = DeepForest.to_json()
with open(logdir + "/model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
DeepForest.save_weights(logdir + "/model.h5")
print("Saved model to disk")

#Calculate confusion and final statistics
#predict
preds, labels=evaluate.predict(DeepForest,test)

#report and log confusion matrix    
tn, fp, fn, tp=evaluate.calculate_confusion(labels,preds,test)
print("True Negative Rate %.3f\nTrue Positive Rate %.3f\nFalse Negative Rate %.3f\nFalse Positive Rate %.3f" % (tn,tp,fn,fp))    

experiment.log_other("True Negative Rate", "{0:.3f}".format(tn))
experiment.log_other("True Positive Rate", "{0:.3f}".format(tp))
experiment.log_other("False Negative Rate", "{0:.3f}".format(fn))
experiment.log_other("False Positive Rate", "{0:.3f}".format(fp))



