'''
Util functions, including dask clients
'''

from keras.callbacks import Callback
from keras import backend as K
import tensorflow as tf
import time

class TimeHistory(Callback):

    def __init__(self,experiment):
        self.experiment = experiment
        
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
        self.experiment.log_other("epoch time", self.times[-1])
        

class PlotImages(Callback):

    def __init__(self,experiment,validation_generator):
        self.experiment = experiment
        self.validation_generator=validation_generator
        
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        
        #Calculate a batch of predictions to view
        images, y_true = next(self.validation_generator) 
        for img in images[:3]:
            self.experiment.log_images("images", img)

