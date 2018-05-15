
from Utilities import *

class Hyperspectral:
    
    def __init__(self,filename):
        self.filename=filename
    
    
    def load(refl_filename):
        '''
        Read a .h5 from file
        '''
        
        self.
        #Read in reflectance hdf5 file (include full or relative path if data is located in a different directory)
        hdf5_file = h5py.File(refl_filename,'r')
        
        
        return reflArray, metadata, wavelengths
    
    def calc_clip_index(clipExtent, h5Extent, xscale=1, yscale=1):
        
        h5rows = h5Extent['yMax'] - h5Extent['yMin']
        h5cols = h5Extent['xMax'] - h5Extent['xMin']    
        
        ind_ext = {}
        ind_ext['xMin'] = round((clipExtent['xMin']-h5Extent['xMin'])/xscale)
        ind_ext['xMax'] = round((clipExtent['xMax']-h5Extent['xMin'])/xscale)
        ind_ext['yMax'] = round(h5rows - (clipExtent['yMin']-h5Extent['yMin'])/xscale)
        ind_ext['yMin'] = round(h5rows - (clipExtent['yMax']-h5Extent['yMin'])/yscale)
        
        return ind_ext
    
    
    def stack_subset_bands(reflArray,reflArray_metadata,bands,clipIndex):
        
        subArray_rows = clipIndex['yMax'] - clipIndex['yMin']
        subArray_cols = clipIndex['xMax'] - clipIndex['xMin']
        
        stackedArray = np.zeros((subArray_rows,subArray_cols,len(bands)),dtype = np.int16)
        band_clean_dict = {}
        band_clean_names = []
        
        for i in range(len(bands)):
            band_clean_names.append("b"+str(bands[i])+"_refl_clean")
            band_clean_dict[band_clean_names[i]] = subset_clean_band(reflArray,reflArray_metadata,clipIndex,bands[i])
            stackedArray[...,i] = band_clean_dict[band_clean_names[i]]
                            
        return stackedArray
    
    
    def subset_clean_band(reflArray,reflArray_metadata,clipIndex,bandIndex):
        
        bandCleaned = reflArray[clipIndex['yMin']:clipIndex['yMax'],clipIndex['xMin']:clipIndex['xMax'],bandIndex].astype(np.int16)
        
        return bandCleaned 
    
    
    def array2raster(newRaster,reflBandArray,reflArray_metadata, extent, ras_dir): 
        
        NP2GDAL_CONVERSION = {
          "uint8": 1,
          "int8": 1,
          "uint16": 2,
          "int16": 3,
          "uint32": 4,
          "int32": 5,
          "float32": 6,
          "float64": 7,
          "complex64": 10,
          "complex128": 11,
        }
        
        pwd = os.getcwd()
        os.chdir(ras_dir) 
        cols = reflBandArray.shape[1]
        rows = reflBandArray.shape[0]
        bands = reflBandArray.shape[2]
        pixelWidth = float(reflArray_metadata['res']['pixelWidth'])
        pixelHeight = -float(reflArray_metadata['res']['pixelHeight'])
        originX = extent['xMin']
        originY = extent['yMax']
        
        driver = gdal.GetDriverByName('GTiff')
        gdaltype = NP2GDAL_CONVERSION[reflBandArray.dtype.name]
        outRaster = driver.Create(newRaster, cols, rows, bands, gdaltype)
        outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
        #outband = outRaster.GetRasterBand(1)
        #outband.WriteArray(reflBandArray[:,:,x])
        for band in range(bands):
            outRaster.GetRasterBand(band+1).WriteArray(reflBandArray[:,:,band])
        
        outRasterSRS = osr.SpatialReference()
        outRasterSRS.ImportFromEPSG(reflArray_metadata['epsg']) 
        outRaster.SetProjection(outRasterSRS.ExportToWkt())
        outRaster.FlushCache()
        os.chdir(pwd) 
    
    def stripe2Raster(f, pt):
        
        full_path = pt+f
        refl, refl_md, wavelengths = load(full_path)
        refl_md['extent']
        
        #Drop water bands
        
        rgb = np.r_[0:425]
        rgb = np.delete(rgb, np.r_[419:426])
        rgb = np.delete(rgb, np.r_[283:315])
        rgb = np.delete(rgb, np.r_[192:210])    
        xmin, xmax, ymin, ymax = refl_md['extent']
        clipExtent = {}
        clipExtent['xMin'] = int(xmin) 
        clipExtent['yMin'] = int(ymin) 
        clipExtent['yMax'] = int(ymax) 
        clipExtent['xMax'] = int(xmax) 
    
    
        subInd = calc_clip_index(clipExtent,refl_md['ext_dict']) 
        subInd['xMax'] = int(subInd['xMax'])
        subInd['xMin'] = int(subInd['xMin'])
        subInd['yMax'] = int(subInd['yMax'])
        subInd['yMin'] = int(subInd['yMin'])
        reflBandArray = stack_subset_bands(refl,refl_md,rgb,subInd)
    
        subArray_rows = subInd['yMax'] - subInd['yMin']
        subArray_cols = subInd['xMax'] - subInd['xMin']
        hcp = np.zeros((subArray_rows,subArray_cols,len(rgb)),dtype = np.int16)
    
        band_clean_dict = {}
        band_clean_names = []
        for i in range(len(rgb)):
            band_clean_names.append("b"+str(rgb[i])+"_refl_clean")
            band_clean_dict[band_clean_names[i]] = refl[:,:,rgb[i]].astype(np.int16)
            hcp[...,i] = band_clean_dict[band_clean_names[i]]
    
        reflArray_metadata = refl_md
        newRaster = f.replace(' ','')[:-3].upper()
        newRaster = str(clipExtent['xMin'])+str((clipExtent['yMax']))+'.tif'
    
        ras_dir = '/ufrc/ewhite/s.marconi/Marconi2018/spatialPhase/rasters/'
        array2raster(newRaster,hcp,reflArray_metadata, clipExtent, ras_dir)

if __name__=="__main__":

    import numpy as np
    import h5py
    import gdal, osr
    import sys
    import ogr, os
    
    #pt = "/ufrc/ewhite/s.marconi/Marconi2018/spatialPhase/OSBS/Reflectance/"
    #f = "NEON_D03_OSBS_DP3_394000_3280000_reflectance.h5"
    #stripe2Raser(f,  pt)
    h=Hyperspectral() 
    h.stripe2Raster(sys.argv[1],  sys.argv[2])

