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

    def __init__(self,experiment):
        self.experiment = experiment
        
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        
        #Calculate predictions
        y_pred = self.model.predict(self.model.validation_data[0])
        self.times.append(time.time() - self.epoch_time_start)
        self.experiment.log_o("epoch time", self.times[-1])

