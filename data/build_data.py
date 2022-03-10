import os
import rasterio
from rasterio import windows
from rasterio.features import rasterize
from osgeo import ogr, gdal
import numpy as np
import geopandas as gpd
from shapely import ops, geometry
from itertools import product

def shp_to_gtiff(path_to_shp, out_path):
    location = os.path.split(path_to_shp)[1].split('.')[0]
    out = out_path+location+'.tif'
    if os.path.exists(out_path+location+'.tif'):
        print("[SKIPPING] GTiff already generated for this SHP.")
        return None
    path_to_rasters = 'data/processed_downloads/'
    for raster in os.listdir(path_to_rasters):
        if raster.startswith(location):
            raster_to_match = path_to_rasters+raster
            break
    image = gdal.Open(raster_to_match, gdal.GA_ReadOnly)
    Shapefile = ogr.Open(path_to_shp)
    Shapefile_layer = Shapefile.GetLayer()
    print("Rasterising shapefile...")
    output = gdal.GetDriverByName('GTiff').Create(out, image.RasterXSize, image.RasterYSize, 1, gdal.GDT_Byte, options=['COMPRESS=DEFLATE'])
    output.SetProjection(image.GetProjectionRef())
    output.SetGeoTransform(image.GetGeoTransform()) 
    Band = output.GetRasterBand(1)
    Band.SetNoDataValue(0)
    gdal.RasterizeLayer(output, [1], Shapefile_layer, burn_values=[1])

    band = None
    output = None
    image = None
    shapefile = None

def get_tiles(ds, width=256, height=256):
    """
    Provide tile window and transform.
    Used by internal methods only, even though access-restriction is not provided.
    """
    nols, nrows = ds.meta['width'], ds.meta['height']
    offsets = product(range(0, nols, width), range(0, nrows, height))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
    for col_off, row_off in  offsets:
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        transform = windows.transform(window, ds.transform)
        yield window, transform

def tile_write(image_path,output_path,size=(256,256)):
    """
    Tile large satellite image and save to specified output location.
    ----- Parameters: -----
    image (opened rasterio dataset object)
    output_path (str) :: path to write to
    size (tuple) :: (height,width) of desired tiles
    """
    
    output_filename = 'tile_{}-{}.tif'
    image = rasterio.open(image_path)
    meta = image.meta.copy()
    
    for window, transform in get_tiles(image,size[0],size[1]):
        print(window)
        meta['transform'] = transform
        if window.width == size[1] and window.height == size[0]:
            meta['width'],meta['height'] = window.width,window.height
            outpath = os.path.join(output_path,output_filename.format(int(window.col_off), int(window.row_off)))
            with rasterio.open(outpath, 'w', **meta) as outds:
                outds.write(image.read(window=window))

if __name__ == '__main__':
    shp = '/localhome/studenter/mikaellv/Project/data/untiled_masks/orkanger_lakes/orkanger_lakes.shp'
    out_path = '/localhome/studenter/mikaellv/Project/data/processed_masks/'
    shp_to_gtiff(path_to_shp=shp,out_path=out_path)
    img = '/localhome/studenter/mikaellv/Project/data/processed_downloads/gaula_hovin_S1A_IW_GRDH_1SDV_20200925T053855.tif'
    out_path = "/localhome/studenter/mikaellv/Project/data/tiled_images/"
    tile_write(img,out_path)
