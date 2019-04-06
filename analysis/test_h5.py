import h5py
import glob
import os
from DeepForest import config

#Load config
DeepForest_config = config.load_config()
pattern=os.path.join(DeepForest_config["TEAK"]["h5"],"*.h5")

files=glob.glob(pattern)

counter=0
for f in files:
    
    try:
        hf = h5py.File(f, 'r')
        shape=hf['train_imgs'][0,].shape
        print("{t} has a shape {s}".format(t=f,s=shape))
    except Exception as e:
        print("{f} failed with error message {e}".format(f=f,e=e))
        counter +=1
        filpath=os.path.splitext(f)[0]
        to_delete_csv=filpath + ".csv"
        try:
            pass
            os.remove(to_delete_csv)
        except Exception as e:
            print(e)
        try: 
            pass
            os.remove(f)
        except Exception as e:
            print(e)

print("deleted {x} corrupt files out of {y} total".format(x=counter,y=len(files)))


import os
import pandas as pd
import glob
import numpy as np
from datetime import datetime
from DeepForest.config import load_config
from DeepForest import preprocess

DeepForest_config = load_config()
data = preprocess.load_csvs(DeepForest_config["training_h5_dir"])

tiles=data.tile.unique()

for f in tiles:
    try:
        tile=os.path.join(DeepForest_config["training_h5_dir"],f) + ".h5"
        hf = h5py.File(tile, 'r')
        shape=hf['train_imgs'][0,].shape
    except:
        to_delete=os.path.join(DeepForest_config["training_h5_dir"],f) + ".csv"
        print("removing {}".format(to_delete))        
        os.remove(to_delete)
        