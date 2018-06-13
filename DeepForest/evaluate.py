'''
Predict DeepForest
'''

from DeepForest.CropGenerator import crop_rgb, data2geojson
from DeepForest.config import config
from DeepForest import preprocess
import argparse
from sklearn.metrics import confusion_matrix
from keras.models import model_from_json
import numpy as np

#Load Model from saved weights, train.py
def load_model(logdir):
    json_file = open(logdir+'/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
   
    # load weights into new model
    loaded_model.load_weights(logdir+"/model.h5")
    print("Loaded model from disk")
    
    return(loaded_model)

#iterate through a list of bounding boxes see ../data, and make predictions
def predict(model,box_file,batch_size):
    
    images=[]
    labels=[]
    
    for id,row in box_file.iterrows():
        
        #Crop and optionally save image
        image=crop_rgb(id, box_file, rgb_tile_dir=config["data_generator_params"]["rgb_tile_dir"])
        images.append(image)
        labels.append(box_file.loc[id].label_numeric)
    
    preds=model.predict(np.array(images),batch_size=batch_size)
    return(preds,labels)
    
# Create visualization
def view_predictions():
    pass
    #Select box
    
    #Expanded crop to give context
    
    #Draw original

## Save image predictions

#Calculate confusion matrix
def calculate_confusion(labels,preds):
    
    tn, fp, fn, tp=confusion_matrix(labels, np.round(preds)).ravel()
    
    #scale metrics
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    tnr=tn/(tn+fp)
        
    return(precision, recall, tnr)
    
    
    
    