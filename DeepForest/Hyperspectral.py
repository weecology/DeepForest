
import numpy as np
import os
import h5py as h5
import matplotlib.pyplot as plt

class Tile:
    
    def __init__(self,filename,site="OSBS"):
        '''
        Read a .h5 from file
        '''        
        
        #On hipergator, need env variable set
        os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'   
        
        #Load and store data as array
        self.filename=filename        
        self.h5file=h5.File(filename,'r')
        
        self.site=site
        
        #get data
        self.data=self.getData()
        
        #get shape
        self.shape=self.getShape()
        
        #get Resolution
        self.res=self.getResolution()
        
        #get coordinates
        self.coords=self.getCoords()
        
        #get wavelength
        self.wavelengths=self.getWavelengths()
        
    def view_items(self):
        self.h5file.visititems(list_dataset)
    
    def getData(self):
        data=self.h5file['OSBS']['Radiance']["Radiance_Data"]
        return(data)
        
    def getShape(self):
        shape=self.h5file[self.site]["Radiance"]["Radiance_Data"].shape
        return(shape)
        
    def getCoords(self):
        '''Get the upper left corner of raster'''
        map_info=self.h5file[self.site]['Radiance']['Metadata']['Coordinate_System']['Map_Info'].value
        xmin=float(str(map_info).split(",")[3])
        
        ymax=float(str(map_info).split(",")[4])   
        
        #find corners
        xmax = xmin + (self.shape[1]*self.res[0])
        ymin = ymax - (self.shape[0]*self.res[1])         
        
        #create extent dictionary
        extDict = {}
        extDict['xmin'] = xmin
        extDict['xmax'] = xmax
        extDict['ymin'] = ymin
        extDict['ymax'] = ymax
        
        return(extDict)
    
    def getResolution(self):
        map_info=str(self.h5file[self.site]['Radiance']['Metadata']['Coordinate_System']['Map_Info'].value).split(",")
        res=float(map_info[5]),float(map_info[6])
        return(res)
        
    def getWavelengths(self):
        
        wavelengths = self.h5file[self.site]['Radiance']['Metadata']['Spectral_Data']['Wavelength']
        print(wavelengths)
        # print(wavelengths.value)
        # Display min & max wavelengths
        print('min wavelength:', np.amin(wavelengths),'nm')
        print('max wavelength:', np.amax(wavelengths),'nm')
        
        # show the band width 
        print('band width =',(wavelengths.value[1]-wavelengths.value[0]),'nm')
        print('band width =',(wavelengths.value[-1]-wavelengths.value[-2]),'nm')
        
        return(wavelengths)
        
    def load_proj4(self):
        return(self.h5file['OSBS']['Radiance']['Metadata']['Coordinate_System']['Proj4'].value)
        
    def extract_band(self,band,clipExtent=None):
        
        '''
        clipExtent: An optional dictionary with xmin,xmax,ymin,ymax
        '''
        
        if not clipExtent:
            
            #No clipping, select band
            b=self.data[:,:,band].astype(np.float)
            
        else:
            #get index for clipping
            sub_Index = calc_clip_index(clipExtent,self.coords)
            
            #extract array
            subArray = self.data[sub_Index['ymin']:sub_Index['ymax'],sub_Index['xmin']:sub_Index['xmax'],:]
            subExt = (clipExtent['xmin'],clipExtent['xmax'],clipExtent['ymin'],clipExtent['ymax'])   
            b = subArray[:,:,band].astype(np.float)        

        #no data value
        #scaleFactor = self.data.attrs['Scale_Factor']
        noDataValue = self.data.attrs['Data_Ignore_Value']
    
        b[b==int(noDataValue)]=np.nan
        #b = b/scaleFactor
        return(b)
    
    def stack_bands(self,bands,clipExtent=None):
        
        '''
        Stack cleaned bands
        '''
        
        if clipExtent:
            rows = clipExtent['ymax'] - clipExtent['ymin']
            cols = clipExtent['xmax'] - clipExtent['xmin']
        else:
            rows=self.shape[0]
            cols=self.shape[1]
        
        stackedArray = np.zeros((rows,cols,len(bands)),'uint8') #pre-allocate stackedArray matrix      
        
        band_clean_dict = {}
        band_clean_names = []
        
        for i in range(len(bands)):
            band_clean_names.append("b"+str(bands[i])+"_refl_clean")
            band_clean_dict[band_clean_names[i]] = self.extract_band(band=bands[i],clipExtent=clipExtent)
            stackedArray[...,i] = band_clean_dict[band_clean_names[i]]*256        
        return(stackedArray)
    
    def NDVI(self,clipExtent,NIR=57,VIS=90):
        '''
        Calculate normalized difference vegetation index
        NIR: Near infrared band
        VIR: Visible band
        '''

        #Check the center wavelengths that these bands represent
        band_width = self.wavelengths.value[1]-self.wavelengths.value[0]
        
        print('Near infrared band is band # %d wavelength range: %.2f - %.2f nm' 
              %(NIR, self.wavelengths.value[NIR]-band_width/2, self.wavelengths.value[NIR]+band_width/2))
        
        print('Visible light band is band # %d wavelength range: %.2f - %.2f nm' 
                   %( VIS, self.wavelengths.value[VIS]-band_width/2, self.wavelengths.value[VIS]+band_width/2))

        #Use the stack_subset_bands function to create a stack of the subsetted red and NIR bands needed to calculate NDVI
        ndvi_stack = self.stack_bands(bands=(NIR,VIS),clipExtent=clipExtent)        

        VIS_data = ndvi_stack[:,:,0].astype(float)
        NIR_data = ndvi_stack[:,:,1].astype(float)
        
        #print if they are equal
        print((VIS_data==NIR_data).all())
        
        #divide if not 0
        a=(NIR_data-VIS_data)
        b=(NIR_data+VIS_data)
        NDVI = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
        return(NDVI)
    
    def plot(self,band_array,colorlimit,title='',cbar ='on',cmap_title='',colormap='spectral',save=False):
        
        '''plot reads in and plots a single band of a reflectance array
        --------
        Parameters
        --------
            band_array: flightline array of reflectance values, created from h5refl2array function
            refl_extent: extent of reflectance data to be plotted (xMin, xMax, yMin, yMax) - use metadata['extent'] from h5refl2array function
            colorlimit: range of values to plot (min,max). Best to look at the histogram of reflectance values before plotting to determine colorlimit.
            ax: optional, default = current axis
            title: string, optional; plot title
            cmap_title: string, optional; colorbar title
            colormap: string, optional; see https://matplotlib.org/examples/color/colormaps_reference.html for list of colormaps
        '''
    
        plot = plt.imshow(band_array); 
        if cbar == 'on':
            cbar = plt.colorbar(plot,aspect=40); plt.set_cmap(colormap); 
            cbar.set_label(cmap_title,rotation=90,labelpad=20)
            
        if save:
            plt.savefig('example.png', bbox_inches='tight')
            
#Util functions
def list_dataset(name,node):
    if isinstance(node, h5.Dataset):
        print(name)

def calc_clip_index(clipExtent, fullExtent, xscale=1, yscale=1):
    
    h5rows = fullExtent['ymax'] - fullExtent['ymin']
    h5cols = fullExtent['xmax'] - fullExtent['xmin']    
    
    indExtent = {}
    indExtent['xmin'] = round((clipExtent['xmin']-fullExtent['xmin'])/xscale)
    indExtent['xmax'] = round((clipExtent['xmax']-fullExtent['xmin'])/xscale)
    indExtent['ymax'] = round(h5rows - (clipExtent['ymin']-fullExtent['ymin'])/xscale)
    indExtent['ymin'] = round(h5rows - (clipExtent['ymax']-fullExtent['ymin'])/yscale)

    return indExtent

if __name__=="__main__":
    
    f=Tile("/orange/ewhite/b.weinstein/NEON/D03/OSBS/DP1.30008.001/2017/FullSite/D03/2017_OSBS_3/L1/Spectrometer/RadianceH5/2017092713_done/NEON_D03_OSBS_DP1_20170927_172515_radiance.h5")
    NDVI=f.NDVI(clipExtent=None)
    f.plot(NDVI,save=True)
    