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
from datetime import datetime
from keras.preprocessing.image import img_to_array
import re
import cv2

###set time
now=datetime.now()
callbacks=keras.callbacks.TensorBoard(log_dir='logs/'+ now.strftime("%Y%m%d-%H%M%S") + '/',write_images=True)

##Load Model
DeepForest=rgb.get_model(is_training=True)

#load data
data=[]
imagePaths=glob.glob("logs/images/*.png")

#load photos

for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (64, 64))
    image = img_to_array(image)
    data.append(image)

#load labels
data = np.array(data, dtype="float") / 255.0

#set labels
pattern=re.compile("background")
is_not_background=[pattern.search(x) is None for x in imagePaths]

s={True: 1, False : 0}
labels=[s[x] for x in is_not_background]

DeepForest.compile(loss="binary_crossentropy",optimizer=keras.optimizers.Adam(), metrics=['acc'])

DeepForest.fit(x=data, y=labels, batch_size=config['training_params']['batch_size'], epochs=1000, verbose=2, callbacks=[callbacks], shuffle=True,validation_split=0.2)

