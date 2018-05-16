'Crop generation and augmentation'

# Authors: Ben Weinstein<ben.weinstein@weecology.org>

import rasterio
from rasterio.tools.mask import mask
from rasterio import plot
from matplotlib import pyplot
import keras

class DataGenerator:
    """
    Generates batches of lidar and rgb data on the fly.
    To be passed as argument in the fit_generator function of Keras.
    """
    
    def __init__(self,list_IDs,batch_size,dim,n_classes=2,shuffle=True):
        
        'Initilization'
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.list_IDs=list_IDs
        self.dim = dim
        self.n_classes=n_classes
        
        #shuffle order
        self.on_epoch_end()               
    
    def __getitem__(self, index):
        'Generate one batch of data, see Keras Sequence method (https://keras.io/utils/)'
        
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def data_gen(self,labels,list_IDs_shuffle):
        """
        Generator to yield batches of lidar and rgb inputs (per sample) along with their labels.
        """
        batch_size = config.batch_size
        while True:
            
            lidar_batch = []
            rgb_batch = []
            batch_labels = []            

            #label array 
            y = np.empty((self.batch_size), dtype=int)
                        
            for id in list_IDs_shuffle:
                
                # Mask
                rgb = crop_rgb(infile, row_id)
                lidar = crop_lidar(infile, row_id)
    
                # Set label
                label = get_label(infile,row_id)
                batch_labels.append(label)
    
                # Pack each input  separately
                lidar_batch.append(top_img)
                rgb_batch.append(bot_img)
                
                #one hot encode labels
                batch_labels=np.array(batch_labels)
                y=keras.utils.to_categorical(batch_labels,num_classes=self.n_classes)
                
            yield [np.array(lidar_batch), np.array(rgb_batch)], y 

            
#Cropping functions             
def crop_lidar(filename,row):
    pass

def crop_rgb(filename,row):
    features=tools.data2geojson(row)
    with rasterio.open(self.filename) as src:
        out_image, out_transform = mask(src, [features], crop=True)
    return(out_image)


## Data augmentations
