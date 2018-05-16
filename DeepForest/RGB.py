'''
RGB data class
Tools for cropping, subseting and viewing orthophoto data using dask
Ben Weinstein ben.weinstein@weecology.oyh
'''

import rasterio
from rasterio.tools.mask import mask
from rasterio import plot
from matplotlib import pyplot

def crop_rgb(filename,row):
    
    with rasterio.open(self.filename) as src:
        out_image, out_transform = mask(src, geoms, crop=True)
        out_meta = src.meta.copy()

out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform})        
else:
            return(out_image)
        
    def save(self):
        '''
        save a raster as a numpy matrix
        '''

