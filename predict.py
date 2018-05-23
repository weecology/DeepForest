'''
Predict DeepForest
'''

from DeepForest.CropGenerator import crop_rgb, data2geojson
from DeepForest.config import config
import matplotlib

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
def predict(model,box_file,box_file):
    
    images=[]
    labels=[]
    
    for id in box_file:
        
        #Crop and optionally save image
        image=crop_rgb(id, box_file, rgb_tile_dir=config["data_generator_params"]["rgb_tile_dir"])
        images.append(image)
        labels.append(box_file.loc[id].label)
    
    preds=model.predict(images,batch_size)
    return(preds)
    
# Create visualization
def view_predictions():
    pass
    #Select box
    
    #Expanded crop to give context
    
    #Draw original

## Save image predictions

#Calculate confusion matrix

if __name__ =="__main__":
    model=load_model(logdir)
    preds=predict(model,box_file)
    