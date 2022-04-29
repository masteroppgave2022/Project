"""
based on http://step.esa.int/docs/tutorials/Performing%20SAR%20processing%20in%20Python%20using%20snappy.pdf
and https://mygeoblog.com/2019/07/08/process-sentinel-1-with-snap-python-api/
"""

import logging
import sre_parse
import numpy as np
import geopandas as gpd
import matplotlib
# matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt 
import matplotlib.colors as colors 
import os
import snappy
from snappy import Product
from snappy import ProductIO 
from snappy import ProductUtils 
from snappy import WKTReader 
from snappy import HashMap
from snappy import GPF
import warnings

# For shapefiles
import shapefile 
import pygeoif

import geopandas as gpd
import rioxarray as rxr
import pandas as pd
from shapely.geometry import Polygon
import shapely
from osgeo import ogr,osr,gdal

class Preprocess():
    def __init__(self, path_to_data="") -> None:
        self.path_to_data = path_to_data

    def read_product(self, name):
        product = ProductIO.readProduct(self.path_to_data+name)
        return product

    def get_product_info(self, product) -> dict:
        width = product.getSceneRasterWidth()
        height = product.getSceneRasterHeight()
        name = product.getName()
        band_names = product.getBandNames()
        print("Band names: {}".format(", ".join(band_names)))

        info = {
            "width": width,
            "height": height,
            "name": name,
            "band_names": "{}".format(", ".join(band_names))
        }

        return info

    def plotBand(self, product, band, vmin, vmax, figname=None):
        print(f"\nplotting {band}...")
        band = product.getBand(band) 
        w = band.getRasterWidth()
        h = band.getRasterHeight() 
        print(w, h)

        band_data = np.zeros(w * h, np.float32) 
        band.readPixels(0, 0, w, h, band_data)

        band_data.shape = h, w

        width = 12
        height = 12
        #plt.figure(figsize=(width, height))
        #imgplot = plt.imshow(band_data, cmap=plt.cm.binary, vmin=vmin, vmax=vmax)
        plt.imshow(band_data, cmap=plt.cm.binary, vmin=vmin, vmax=vmax)
        plt.show()
        #if plt.figimage != None:
        #    plt.savefig(figname)
        #else:
        #    plt.show(imgplot)

        #return imgplot

    def apply_orbit_file(self, product):
        parameters = HashMap() 
        GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()
        parameters.put('Apply-Orbit-File', True)
        apply_orbit_file = GPF.createProduct('Apply-Orbit-File', parameters, product)
        return apply_orbit_file

    def add_shape_file(self, product, path_to_shapefile):
        r = shapefile.Reader(path_to_shapefile)
        g=[]
        for s in r.shapes(): 
            g.append(pygeoif.geometry.as_shape(s))
        m = pygeoif.MultiPoint(g)
        wkt = str(m.wkt).replace("MULTIPOINT", "POLYGON(") + ")"
        SubsetOp = snappy.jpy.get_type('org.esa.snap.core.gpf.common.SubsetOp') 
        bounding_wkt = wkt
        geometry = WKTReader().read(bounding_wkt)
        HashMap = snappy.jpy.get_type('java.util.HashMap') 
        GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis() 
        parameters = HashMap()
        parameters.put('copyMetadata', True)
        parameters.put('geoRegion', geometry)
        product_subset = snappy.GPF.createProduct('Subset', parameters, product)
        return product_subset

    def apply_thermal_noise_removal(self, product):
        parameters = HashMap()
        parameters.put('removeThermalNoise', True)
        output = GPF.createProduct('ThermalNoiseRemoval', parameters, product)
        return output

    def calibrate(self, product):
        parameters = HashMap() 
        parameters.put('outputSigmaBand', True)
        parameters.put('sourceBands', 'Intensity_VH,Intensity_VV') 
        parameters.put('selectedPolarisations', 'VH,VV')
        parameters.put('outputImageScaleInDb', False)
        product_calibrated = GPF.createProduct("Calibration", parameters, product)
        return product_calibrated
        
    def multilook(self, product):
        print("\napplying mulilooking")
        azLooks = 3
        rgLooks = 3
        parameters = HashMap()
        parameters.put('grSquarePixel', True)
        parameters.put('nRgLooks', rgLooks)
        parameters.put('nAzLooks', azLooks)
        parameters.put('outputIntensity', False)
        product_multilooked = snappy.GPF.createProduct("Multilook", parameters, product)
        return product_multilooked


    def speckle_filter(self, product):
        parameters = HashMap()
        parameters.put('sourceBands', 'Sigma0_VV,Sigma0_VH') 
        parameters.put('filter', 'Frost') 
        parameters.put('filterSizeX', '5') 
        parameters.put('filterSizeY', '5')
        speckle_filter = snappy.GPF.createProduct('Speckle-Filter', parameters, product)
        return speckle_filter

    def terrain_correction(self, product):
        parameters = HashMap() 
        parameters.put('saveSelectedSourceBand', True)
        parameters.put('demName', 'External DEM')
        parameters.put('externalDEMFile', '/localhome/studenter/mikaellv/Project/data/DEM_merge/DEM_merge.tif')
        parameters.put('pixelSpacingInMeter', 10.0)
        parameters.put('nodataValueAtSea', False)
        terrain_corrected = GPF.createProduct("Terrain-Correction", parameters, product)
        return terrain_corrected
    
    def add_elevation_band(self, product):
        parameters = HashMap()
        parameters.put('demName', 'External DEM')
        parameters.put('externalDEMFile', '/localhome/studenter/mikaellv/Project/data/DEM_merge/DEM_merge.tif')
        added_elevation = GPF.createProduct("AddElevation", parameters, product)
        return added_elevation
    
    def save_product(self, product, name, path, type="GeoTIFF"):
        """ Type = "BEAM-DIMAP" for snap, else "GeoTIFF" """
        ProductIO.writeProduct(product, path+name, type)

    def geopos_to_wkt(self, geopos):
        lat = []
        long =[]
        for e in geopos:
            lat.append(e.lat)
            long.append(e.lon)
        polygon_geom = Polygon(zip(long, lat))
        polygon = gpd.GeoDataFrame(crs='epsg:4326', geometry=[polygon_geom])       
        return polygon
    
    def subset(self, product, shape):
        """
        Type = "BEAM-DIMAP" for snap, else "GeoTIFF"
        """
        GeoPos = snappy.ProductUtils.createGeoBoundary(product, 1)
        scene = self.geopos_to_wkt(GeoPos)
        r = shapefile.Reader(shape)
        g=[]
        for s in r.shapes(): 
            g.append(pygeoif.geometry.as_shape(s))
        m = pygeoif.MultiPoint(g)
        wkt = str(m.wkt).replace("MULTIPOINT", "POLYGON(") + ")"
        shape_wkt = shapely.wkt.loads(wkt)
        contains = scene.contains(shape_wkt)
        SubsetOp = snappy.jpy.get_type('org.esa.snap.core.gpf.common.SubsetOp') 
        bounding_wkt = wkt
        geometry = WKTReader().read(bounding_wkt)
        HashMap = snappy.jpy.get_type('java.util.HashMap') 
        GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis() 
        parameters = HashMap()
        parameters.put('copyMetadata', True)
        parameters.put('geoRegion', geometry)
        if contains.bool():
            product_subset = snappy.GPF.createProduct('Subset', parameters, product)
            return product_subset
        else: return None

    def clip_shapefile(self, source_shp, mask_shps, destination):
        print("[INFO] Reading FKB_vann ...")
        src = gpd.read_file(source_shp)
        print("[INFO] Done reading, proceeding to clip ...")
        for mask in mask_shps:
            print(f"[INFO] Clipping {os.path.split(mask)[1]}")
            output_name = os.path.split(mask)[1]
            out_path = destination+output_name.split('.')[0]+'/'
            if not os.path.exists(out_path):
                os.makedirs(out_path)
                out_path = out_path+output_name
                extent = gpd.read_file(mask)
            else:
                print("[SKIPPING] ... Clipped SHP already exists.")
                continue
                # Check CRS:
            if not src.crs == extent.crs:
                print(f"[SKIPPING] CRS of source SHP and mask SHP: {mask} are not the same.")
                continue
            clipped = gpd.clip(src, extent)
            print("[INFO] Done clipping, checking for invalid geometries in dataframe ... ")
            for i,row in clipped.iterrows():
                if (type(row.geometry) == shapely.geometry.collection.GeometryCollection)\
                    or (type(row.geometry) == shapely.geometry.multilinestring.MultiLineString):
                    shapes = []
                    for shape in row.geometry:
                        if type(shape) == shapely.geometry.polygon.Polygon: shapes.append(shape)
                    clipped.at[i, 'geometry'] = shapely.geometry.collection.GeometryCollection(shapes)
            clipped.to_file(out_path)
            print(f"[INFO] 100% done with: {os.path.split(mask)[1]}!")

    def clip_raster(self, gtiff_raster, shp, out_path, name):
        raster = rxr.open_rasterio(gtiff_raster,masked=True).squeeze()
        shape = gpd.read_file(shp)
        clipped = raster.rio.clip(shape.geometry.apply(shapely.geometry.mapping),shape.crs)
        clipped.rio.to_raster(out_path+'/'+name+'.tif')



                
            
        



if __name__=='__main__':
    """ Just for testing purposes: """
    # prosess = Preprocess()
    
    # product = prosess.read_product("unprocessed_downloads/S1B_IW_GRDH_1SDV_20200910T060300_20200910T060325_023309_02C443_0BCF.zip")

    # info = prosess.get_product_info(product)

    # print(info)
    # prosess.plotBand(product, "Intensity_VV", 0, 100000, "testimage2.png")

    # subset = prosess.add_shape_file(product,"shapefiles/molde/molde.shp")
    # prosess.plotBand(subset, "Intensity_VV", 0, 100000) #"subset2.png"

    # product = prosess.apply_orbit_file(product)
    # prosess.plotBand(product, "Intensity_VV", 0, 100000, "testimage_orbit2.png")

    # product = prosess.apply_thermal_noise_removal(product)
    # prosess.plotBand(product, "Intensity_VV", 0, 100000, "testimage_thermalnoise2.png")

    # product = prosess.calibrate(product)
    # prosess.plotBand(product, "Beta0_VV", 0, 1, "testimage_calibrate2.png")

    # product = prosess.speckle_filter(product)
    # prosess.plotBand(product, "Beta0_VV", 0, 1, "testimage_speckle2.png")

    # info = prosess.get_product_info(product)

    # print(info)

    # #product = prosess.terrain_flattening(product)
    # #prosess.plotBand(product, "Gamma0_VV", 0, 0.1, "testimage_terrainflattened2.png")

    # product = prosess.terrain_correction(product)
    # prosess.plotBand(product, "Beta0_VV", 0, 0.1, "testimage_terraincorrected2.png")

    # info = prosess.get_product_info(product)

    # print(info)

    # subset = prosess.add_shape_file(product,"shapefiles/molde2/mol2.shp")
    # prosess.plotBand(subset, "Beta0_VV", 0, 0.1, "subset222.png")

    # subset = prosess.add_shape_file(product,"shapefiles/molde/molde.shp")
    # prosess.plotBand(subset, "Beta0_VV", 0, 0.1, "subset22.png")

    # """ Testing shp clipping method: """
    # pp = Preprocess()
    # src = '/localhome/studenter/mikaellv/Project/data/FKB_vann/FKB_vann.shp'
    # masks = ['/localhome/studenter/mikaellv/Project/data/shapefiles/surnadal_lakes/surnadal_lakes.shp']
    # dest = '/localhome/studenter/mikaellv/Project/data/untiled_masks/'
    # pp.clip_shapefile(src,masks,dest)

    """ Testing raster clipping """
    pp = Preprocess()
    raster = '/localhome/studenter/mikaellv/Project/data/processed_downloads/melhus_lakes_S1B_IW_GRDH_1SDV_20200627T053825_20200627T053850_022215_02A297_D3A3.zip.dim.tif'
    shp = '/localhome/studenter/mikaellv/Project/data/shapefiles/melhus_lakes/melhus_lakes.shp'
    pp.clip_raster(raster, shp)
    