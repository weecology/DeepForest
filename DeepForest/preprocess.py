'''
Preprocess data
Loading, Standardizing and Filtering the raw data to dimish false positive labels 
'''

import pandas as pd
import glob
from .config import config
import os

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


def load_data(data_dir=config['bbox_data_dir'],nsamples=config["subsample"]):
    '''
    data_dir: path to .csv files. Optionall can be a path to a specific .csv file.
    nsamples: Number of total samples, "all" will yield full dataset
    '''
    
    if(os.path.splitext(data_dir)[-1]==".csv"):
        data=pd.read_csv(data_dir,index_col=0)
    else:
        #Gather data
        data_paths=glob.glob(data_dir+"/*.csv")
        dataframes = (pd.read_csv(f,index_col=0) for f in data_paths)
        data = pd.concat(dataframes, ignore_index=True)
        
    #set index explicitely
    data=data.set_index('box')
    
    #create numeric labels
    lookup={"Background": 0, "Tree": 1}
    data['label_numeric']=[lookup[x] for x in data.label]    
    
    #optionally subset data, if config argument is numeric, subset data
    if(not isinstance(nsamples,str)):
        data=data.sample(n=nsamples, random_state=2)
        
    return(data)
    
def zero_area(data):
    data=data[data.xmin!=data.xmax]    
    return(data)

#Allometry of height to tree size
def allometry(data):
    pass


#Filter by ndvi threshold
def ndvi(data):
    pass

