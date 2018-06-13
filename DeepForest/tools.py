'''
Util functions, including dask clients
'''

from keras.callbacks import Callback
from keras import backend as K
import tensorflow as tf
import time

##Keras Callbacks

class TimeHistory(Callback):

    def __init__(self,experiment,nsamples):
        self.experiment = experiment
        self.nsamples=nsamples
        
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
        
        #samples per second
        seconds_per_sample=self.times[-1]/self.nsamples
        self.experiment.log_other("seconds_per_sample", seconds_per_sample)


