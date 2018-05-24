'Crop generation and augmentation for keras model fitting'

# Authors: Ben Weinstein<ben.weinstein@weecology.org>

import rasterio
from rasterio.tools.mask import mask
from rasterio import plot
import numpy as np
import cv2
import keras

class DataGenerator(keras.utils.Sequence):
    """
    Generates batches of lidar and rgb data on the fly.
    To be passed as argument in the fit_generator function of Keras.
    """
    
    def __init__(self,box_file,list_IDs,labels,batch_size,rgb_tile_dir,n_classes=2,shuffle=True):
        
        'Initilization'
        self.batch_size=batch_size
        self.rgb_tile_dir=rgb_tile_dir
        self.labels=labels
        self.shuffle=shuffle
        self.list_IDs=list_IDs
        self.n_classes=n_classes
        self.box_file=box_file
        
        #shuffle order
        self.on_epoch_end()               
    
    def __getitem__(self, index):
        'Generate one batch of data, see Keras Sequence method (https://keras.io/utils/)'
        
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(labels=self.labels,list_IDs_shuffle=list_IDs_temp)

        return X, y
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self,labels,list_IDs_shuffle):
        """
        Generator to yield batches of lidar and rgb inputs (per sample) along with their labels.
        """
        
        while True:
            
            rgb_batch = []
            batch_labels = []            

            #label array 
            #TODO preset numpy array size? Faster.
                        
            for id in list_IDs_shuffle:
                
                # Mask
                rgb = crop_rgb(id,self.box_file,self.rgb_tile_dir,show=False)
    
                # Pack each input  separately
                rgb_batch.append(rgb)
                
                #one hot encode labels
                label = labels[id]
                batch_labels.append(label)
                
            return np.array(rgb_batch), np.array(batch_labels)

            
### Cropping functions###
#####################
            
#RGB

def crop_rgb(id,file,rgb_tile_dir,show=False):
    
    #select row
    row=file.loc[id]
    
    #create polygon from bounding box
    features=data2geojson(row)
        
    #crop and return image
    with rasterio.open(rgb_tile_dir + row.rgb_path) as src:
        out_image, out_transform = mask(src, [features], crop=True)
        
    #color channel should be last
    out_image=np.moveaxis(out_image, 0, -1)     
        
    if show:
        #cv2 takes bgr order
        to_write=out_image[...,::-1]
        cv2.imwrite("logs/images/" + id + ".png",to_write)    

    #resize image and rescale
    image_resize=cv2.resize(out_image,(150,150))
    image_rescale=image_resize/255.0
    return(image_rescale)        
    
def data2geojson(row):
    '''
    Convert a pandas row into a polygon bounding box
    ''' 
    
    features={"type": "Polygon",
         "coordinates": 
         [[(float(row["xmin"]),float(row["ymin"])),
             (float(row["xmax"]),float(row["ymin"])),
             (float(row["xmax"]),float(row["ymax"])),
             (float(row["xmin"]),float(row["ymax"])),
             (float(row["xmin"]),float(row["ymin"]))]]}       
    
    return features

if __name__ =="__main__":
    pass