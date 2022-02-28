from configparser import Interpolation
import os
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

import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
import math

def plot(image_path, mask_path):
    img_arr = rasterio.open(image_path).read()#*10
    image = np.rollaxis(img_arr,0,3)
    shp = gpd.read_file(mask_path)
    ext = shp.total_bounds
    extent = [ext[0],ext[2],ext[1],ext[3]]
    fig, ax = plt.subplots(figsize=(25,25))
    plt.tight_layout()
    ax.imshow(image,extent=extent)
    shp.plot(ax=ax,facecolor='green',edgecolor='none',zorder=5,alpha=0.2)
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
            for a in range(len(e)):
                if math.isnan(e[a]): e[a]=0

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
    print(image[1])
    fig, axs = plt.subplots(1, 3, figsize=(25,25))
    plt.tight_layout()
    #axs[0].imshow(image[:, :, 2], cmap='hot', interpolation="nearest") #image[:, :, 1]
    axs[0].hist(image.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
    axs[0].set_title('Original Image')
    axs[2].imshow(mask)
    axs[2].set_title('Segmentation Mask')
    axs[1].imshow(image)
    axs[1].imshow(mask, alpha=0.5)
    axs[1].set_title('Masked Image')
    plt.show()




if __name__=='__main__':
     
    root_images = '/localhome/studenter/mikaellv/Project/data/processed_downloads/'
    root_masks = '/localhome/studenter/mikaellv/Project/data/untiled_masks/'
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
        mask_path = root_masks+loc+'/'+loc+'.shp'
        shp = gpd.read_file(mask_path)

        data = rxr.open_rasterio(img_path, masked=True)
        data_plotting_extent = plotting_extent(data[0], data.rio.transform())




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
        
        """
        break
    image, mask = LoadImage('/localhome/studenter/renatask/Project/data/processed_downloads/andalsnes_S1A_IW_GRDH_1SDV_20200227T170317.tif', '/localhome/studenter/renatask/Result.tif')
    plotMaskedImage(image, mask)

        

        