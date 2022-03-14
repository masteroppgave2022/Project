from configparser import Interpolation
import os
from black import TRANSFORMED_MAGICS
from matplotlib.colors import Normalize
#from cv2 import norm
import numpy as np
from pyparsing import alphas
import rasterio
import geopandas as gpd
import rioxarray as rxr
from rasterio.plot import show
from rasterio.plot import plotting_extent
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt

# import earthpy as et
# import earthpy.spatial as es
# import earthpy.plot as ep
import math

# import georaster
from osgeo import gdal


def plot(image_path, mask_path):
    img_arr = rasterio.open(image_path).read()#*10
    image = np.rollaxis(img_arr,0,3)
    shp = gpd.read_file(mask_path)
    ext = shp.total_bounds
    extent = [ext[0],ext[2],ext[1],ext[3]]
    fig, ax = plt.subplots(figsize=(25,25))
    plt.tight_layout()
    ax.imshow(image,extent=extent)
    shp.plot(ax=ax,facecolor='green',edgecolor='none',zorder=5,alpha=0.1)
    plt.show()

def LoadImage(image_path, mask_path):
    """ Return: (h,w,n)-np.arrays """
    # Images to np-arrays
    image_arr = rasterio.open(image_path).read()
    mask_arr = rasterio.open(mask_path).read()
    # Convert dimensions to standard (n,height,width) --> (height,width,n)
    image = np.rollaxis(image_arr,0,3)
    mask = np.rollaxis(mask_arr,0,3)

    for i in image:
        for e in i:
        #for e in range(len(i)):
    #        if math.isnan(i[e]): i[e]=0
            for a in range(len(e)): 
                if math.isnan(e[a]): e[a] = 0
    #image = scale(image)
    return image, mask


def scale(x, out_range=(0, 255)):
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2

def plotMaskedImage(image, mask):
    #for i in image:
    #    for e in i:
    #        for a in range(len(e)):
    #            if math.isnan(e[a]): e[a]=0
    #image = scale(image)
    #print(image[1])
    fig, axs = plt.subplots(1, 3, figsize=(25,25))
    plt.tight_layout()
    axs[0].imshow(10*np.log10(image[:, :, 0]), cmap='ocean') #image[:, :, 1]
    #axs[0].hist(image.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
    #rasterio.plot.show(image, ax=axs[0], adjust='linear')
    axs[0].set_title('Original Image')
    axs[2].imshow(mask[:, :, 0])
    axs[2].set_title('Segmentation Mask')
    axs[1].imshow(mask[:, :, 0])
    axs[1].imshow(10*np.log10(image[:, :, 0]), cmap='ocean', alpha=0.6)
    #rasterio.plot.show(image, ax=axs[1], adjust=False)
    #axs[1].imshow(mask, alpha=0.5)
    axs[1].set_title('Masked Image')
    plt.show()

def plotPred(image, mask, pred):
    #for i in image:
    #    for e in i:
    #        for a in range(len(e)):
    #            if math.isnan(e[a]): e[a]=0
    #image = scale(image)
    #print(image[1])
    fig, axs = plt.subplots(1, 3, figsize=(25,25))
    plt.tight_layout()
    axs[0].imshow(10*np.log10(image[:, :, 0]), cmap='ocean') #image[:, :, 1]
    #axs[0].hist(image.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
    #rasterio.plot.show(image, ax=axs[0], adjust='linear')
    axs[0].set_title('Original Image')
    axs[2].imshow(mask[:, :, 1])
    axs[2].set_title('Segmentation Mask')
    axs[1].imshow(pred[:, :, 0])
    #axs[1].imshow(10*np.log10(image[:, :, 0]), cmap='ocean', alpha=0.6)
    #rasterio.plot.show(image, ax=axs[1], adjust=False)
    #axs[1].imshow(mask, alpha=0.5)
    axs[1].set_title('Predicted mask')
    plt.show()



if __name__=='__main__':
     
    root_images = '/localhome/studenter/mikaellv/Project/data/processed_downloads/'
    root_masks = '/localhome/studenter/renatask/Project/data/processed_masks/'
    """
    for img in os.listdir(root_images):
        loc = img.split('S')[0][:-1]
        img_path = root_images+img
        mask_path = root_masks+loc+'/'+loc+'.shp'
        img_arr = rasterio.open(img_path).read()#*10
        image = np.rollaxis(img_arr,0,3)
        shp = gpd.read_file(mask_path)
        ext = shp.total_bounds
        extent = [ext[0],ext[2],ext[1],ext[3]]
        fig, ax = plt.subplots(figsize=(25,25))
        plt.tight_layout()
        ax.imshow(image,extent=extent)
        shp.plot(ax=ax,facecolor='green',edgecolor='none',zorder=5,alpha=0.2)
        plt.show()
    """
    for img in os.listdir(root_images):
        loc = img.split('S')[0][:-1]
        img_path = root_images+img
        mask_path = root_masks+loc+'.tif'
        # shp = gpd.read_file(mask_path)

        # data = rxr.open_rasterio(img_path, masked=True)
        # data_plotting_extent = plotting_extent(data[0], data.rio.transform())
        img, mask = LoadImage(img_path, mask_path)
        plotMaskedImage(img, mask)




        """ 
        f, ax = plt.subplots()

        ep.plot_rgb(data.values,
                    rgb=[1, 2, 3],
                    ax=ax,
                    title="",
                    extent=data_plotting_extent)  # Use plotting extent from DatasetReader object

        shp.plot(ax=ax)

        plt.show()

        with rasterio.open(img_path) as src:
            ep.plot_bands(src.read()*10,
                       title='',
                       figsize=(8, 3))
        plt.show()

        with rasterio.open(img_path) as src:
            dem = src.read()
            dem = np.rollaxis(dem,0,3)
            fig, ax = plt.subplots(figsize = (25, 25))
        im = ax.imshow(dem.squeeze())
        ep.colorbar(im)
        plt.show

        # dataset = gdal.Open(path, gdal.GA_ReadOnly) 
        # # Note GetRasterBand() takes band no. starting from 1 not 0
        # band = dataset.GetRasterBand(1)
        # arr = band.ReadAsArray()
        # plt.imshow(arr/1.73)
        # print(band.GetStatistics(True, True))
        """
        #break
    path = '/localhome/studenter/renatask/Project/data/processed_downloads/andalsnes_S1A_IW_GRDH_1SDV_20200227T170317.tif'
    img = georaster.MultiBandRaster('/localhome/studenter/renatask/Project/data/processed_downloads/andalsnes_S1A_IW_GRDH_1SDV_20200227T170317.tif')
    # img.plot()
    # plt.imshow(img.r)


    image, mask = LoadImage('/localhome/studenter/renatask/Project/data/processed_downloads/andalsnes_S1A_IW_GRDH_1SDV_20200227T170317.tif', '/localhome/studenter/renatask/Result.tif')
    plotMaskedImage(image, mask)

    with rasterio.open(path) as ds: 
        backscatter = ds.read() 
        backscatter_profile = ds.profile

    print(backscatter_profile)
    print('This is the crs: ', backscatter_profile['crs'])
    print('This is the origin and resolution data: ', backscatter_profile['transform'])
    print('This is the datatype of the raster: ', backscatter_profile['dtype'])
    print('This is how many bands are in the raster', backscatter_profile['count'])
    
    #plt.imshow(backscatter[0, ...])
    plt.imshow(10*np.log10(backscatter[1,...]))
    plt.colorbar()
    #plt.show()


        

        