import os
import numpy as np
import rasterio
import geopandas as gpd
import rioxarray as rxr
from rasterio.plot import show
from rasterio.plot import plotting_extent
from matplotlib import pyplot as plt


if __name__=='__main__':
    root_images = '/localhome/studenter/mikaellv/Project/data/processed_downloads/'
    root_masks = '/localhome/studenter/mikaellv/Project/data/untiled_masks/'
    for img in os.listdir(root_images):
        loc = img.split('S')[0][:-1]
        img_path = root_images+img
        mask_path = root_masks+loc+'/'+loc+'.shp'
        img_arr = rasterio.open(img_path).read()
        image = np.rollaxis(img_arr,0,3)
        shp = gpd.read_file(mask_path)
        ext = shp.total_bounds
        extent = [ext[0],ext[2],ext[1],ext[3]]
        fig, ax = plt.subplots(figsize=(25,25))
        plt.tight_layout()
        ax.imshow(image,extent=extent)
        shp.plot(ax=ax,facecolor='green',edgecolor='none',zorder=5,alpha=0.2)
        plt.show()


        