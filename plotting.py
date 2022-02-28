
import rasterio
import numpy as np
from rasterio import plot as rasterplot
import geopandas as gpd
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt

"""

# this is how you'd open the raster dataset if you have one
tiff = rasterio.open('/localhome/studenter/renatask/Project/data/processed_downloads/andalsnes_S1A_IW_GRDH_1SDV_20200227T170317.tif')
tiff_extent = [tiff.bounds[0], tiff.bounds[2], tiff.bounds[1], tiff.bounds[3]]

# i am making this array up
#tiff_band_1 = np.random.randint(0, 10, size=(65, 64))
#tiff_extent = [4159200.0, 4808100.0, 2828000.0, 3482600.0]

shapefile = gpd.read_file('/localhome/studenter/renatask/Project/data/untiled_masks/andalsnes/andalsnes.shp')
#shapefile = shapefile.to_crs('epsg:3035')
#shapefile = shapefile[shapefile.name == 'Germany']

f, ax = plt.subplots()

# plot DEM
rasterplot.show(
    tiff.read(1),  # use tiff.read(1) with your data
    extent=tiff_extent,
    ax=ax,
    adjust='linear',
    

)
# plot shapefiles
shapefile.plot(ax=ax, facecolor='w', edgecolor='k')
plt.show() 
"""

# A script to rasterise a shapefile to the same projection & pixel resolution as a reference image.
from osgeo import ogr, gdal
import subprocess

InputVector = '/localhome/studenter/renatask/Project/data/untiled_masks/andalsnes/andalsnes.shp'
OutputImage = 'Result.tif'

RefImage = '/localhome/studenter/renatask/Project/data/processed_downloads/andalsnes_S1A_IW_GRDH_1SDV_20200227T170317.tif'

gdalformat = 'GTiff'
datatype = gdal.GDT_Byte
burnVal = 1 #value for the output image pixels
##########################################################
# Get projection info from reference image
Image = gdal.Open(RefImage, gdal.GA_ReadOnly)

# Open Shapefile
Shapefile = ogr.Open(InputVector)
Shapefile_layer = Shapefile.GetLayer()

# Rasterise
print("Rasterising shapefile...")
Output = gdal.GetDriverByName(gdalformat).Create(OutputImage, Image.RasterXSize, Image.RasterYSize, 1, datatype, options=['COMPRESS=DEFLATE'])
Output.SetProjection(Image.GetProjectionRef())
Output.SetGeoTransform(Image.GetGeoTransform()) 

# Write data to band 1
Band = Output.GetRasterBand(1)
Band.SetNoDataValue(0)
gdal.RasterizeLayer(Output, [1], Shapefile_layer, burn_values=[burnVal])

# Close datasets
Band = None
Output = None
Image = None
Shapefile = None

# Build image overviews
subprocess.call("gdaladdo --config COMPRESS_OVERVIEW DEFLATE "+OutputImage+" 2 4 8 16 32 64", shell=True)
print("Done.")