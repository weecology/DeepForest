'Crop generation and augmentation for keras model fitting'

# Authors: Ben Weinstein<ben.weinstein@weecology.org>

import rasterio
from rasterio.tools.mask import mask
from rasterio import plot
from matplotlib import pyplot
import numpy as np
from cv2 import resize
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
            
            #lidar_batch = []
            rgb_batch = []
            batch_labels = []            

            #label array 
            #TODO preset numpy array size? Faster
                        
            for id in list_IDs_shuffle:
                print(id)
                # Mask
                rgb = crop_rgb(id,self.box_file,self.rgb_tile_dir)
                #lidar = crop_lidar(id,self.box_file)
    
                # Pack each input  separately
                #lidar_batch.append(rgb)
                rgb_batch.append(rgb)
                
                #one hot encode labels
                label = self.box_file.loc[id].label
                batch_labels.append(label)
                
            y=keras.utils.to_categorical(batch_labels,num_classes=self.n_classes)
                
            yield [np.array(rgb_batch)], y 

            
### Cropping functions###
#####################
            
#RGB

def crop_rgb(id,file,rgb_tile_dir):
    
    #select row
    row=file.loc[id]
    
    #create polygon from bounding box
    features=data2geojson(row)
    
    #crop and return image
    with rasterio.open(rgb_tile_dir + row.rgb_path) as src:
        out_image, out_transform = mask(src, [features], crop=True)
        
    #color channel should be last, check the order of rasterio, all should be rotation invariant?
    out_image=np.moveaxis(out_image, 0, -1)

    #TODO resize crop
    image_resize=resize(out_image.data,(32,32))
    
    return(image_resize)        
    
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

#Lidar

def crop_lidar(id,file):
    #select row
    row=file.loc(id)
    
    return(lidar_points)            



## Data augmentations
