'''
Preprocess data
Loading, Standardizing and Filtering the raw data to dimish false positive labels 
'''

import pandas as pd
import glob
from .config import config
import os
import random

def load_data(data_dir):
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
        data = pd.concat(dataframes, ignore_index=False)
        
    return(data)
    
def zero_area(data):
    data=data[data.xmin!=data.xmax]    
    return(data)

#Allometry of height to tree size
def allometry(data):
    pass


#Filter by ndvi threshold
def NDVI(data,threshold,data_dir):
    
    #for each row
    for row,index in data.iterrows():
        
        #create the hyperspectral object
        h=Hyperspectral(data_dir + row['hyperspec_path'])
        
        #create clipExtent from box
        clipExtent={}
        clipExtent["xmin"]=row["xmin"]
        clipExtent["ymin"]=row["ymin"]
        clipExtent["xmax"]=row["xmax"]
        clipExtent["ymax"]=row["ymax"]
        
        #Calculate NDVI
        NDVI=f.NDVI(clipExtent=clipExtent)
        
        data['NDVI']=NDVI
    
    #Create lower bound for NDVI   
    data=data[data.NDVI > threshold]    
    return(data)
