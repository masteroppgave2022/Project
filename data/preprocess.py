"""
based on http://step.esa.int/docs/tutorials/Performing%20SAR%20processing%20in%20Python%20using%20snappy.pdf
"""
import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt 
import matplotlib.colors as colors 
import os
import snappy

# For shapefiles
import shapefile 
import pygeoif

class Preprocess():
    def __init__(self, path_to_data) -> None:
        self.path_to_data = path_to_data

    def read_product(self, name):
        product = snappy.ProductIO.readProduct(self.path_to_data+name)
        return product

    def get_product_info(self, product) -> dict:
        width = product.getSceneRasterWidth()
        height = product.getSceneRasterHeight()
        name = product.getName()
        band_names = product.getBandNames()
        info = pd.DataFrame({
            "width": [width],
            "height": [height],
            "name": [name],
            "band_names": ["{}".format(", ".join(band_names))]
        })
        return info

    def create_subset(self, product, x, y, width, height):
        """
        Just to achieve a smaller image for the purpose of speedier test-processing.

        (x,y) = (0,0) is reference in upper right corner of raster.
        """
        parameters = snappy.HashMap()
        parameters.put('copyMetadata', True)
        parameters.put('region', "%s,%s,%s,%s" % (x,y,width,height))
        subset = snappy.GPF.createProduct('Subset', parameters, product)
        return subset

    def plot_band(self, product, band, vmin, vmax, save_path=None):
        band = product.getBand(band) 
        w = band.getRasterWidth()
        h = band.getRasterHeight() 
        band_data = np.zeros(w * h, np.float32) 
        band.readPixels(0, 0, w, h, band_data)
        print(max(band_data))
        print(min(band_data))
        band_data.shape = h, w
        plt.figure(figsize=(12, 12))
        imgplot = plt.imshow(band_data, cmap=plt.cm.binary, vmin=vmin, vmax=vmax)
        # plt.imshow(band_data, cmap=plt.cm.binary, vmin=vmin, vmax=vmax)
        if plt.figimage != None:
            plt.savefig(save_path)
        else:
            plt.show(imgplot)

        return imgplot

    def apply_orbit_file(self, product):
        parameters = snappy.HashMap() 
        snappy.GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()
        parameters.put('Apply-Orbit-File', True)
        parameters.put('orbitType', 'Sentinel Precise (Auto Download)') 
        parameters.put('polyDegree', '3') 
        parameters.put('continueOnFail', 'false')

        apply_orbit_file = snappy.GPF.createProduct('Apply-Orbit-File', parameters, product)
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

        geometry = snappy.WKTReader().read(bounding_wkt)

        HashMap = snappy.jpy.get_type('java.util.HashMap') 
        snappy.GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis() 
        parameters = HashMap()
        parameters.put('copyMetadata', True)
        parameters.put('geoRegion', geometry)
        product_subset = snappy.GPF.createProduct('Subset', parameters, product)

        return product_subset

    def thermal_noise_removal(self, product):
        parameters = snappy.HashMap()
        parameters.put('removeThermalNoise', True)
        therm_noise_corrected = snappy.GPF.createProduct('ThermalNoiseRemoval', parameters, product)
        return therm_noise_corrected

    def calibrate(self, product):
        parameters = snappy.HashMap() 
        parameters.put('outputSigmaBand', True) 
        parameters.put('sourceBands', 'Intensity_VH,Intensity_VV') 
        parameters.put('selectedPolarisations', 'VH,VV') 
        parameters.put('outputImageScaleInDb', False)
        product_calibrated = snappy.GPF.createProduct("Calibration", parameters, product)

        return product_calibrated

    def speckle_filter(self, product):
        """
        TODO: check values
        """
        parameters = snappy.HashMap()
        # parameters.put('sourceBands', 'Sigma0_VH') 
        parameters.put('filter', 'Lee Sigma') # Lee filters are edge-preserving, so probably worth keeping, testing with Lee Sigma
        parameters.put('filterSizeX', 5) 
        parameters.put('filterSizeY', 5) 
        # parameters.put('dampingFactor', '2') 
        # parameters.put('estimateENL', 'true') 
        # parameters.put('enl', '1.0') 
        # parameters.put('numLooksStr', '1') 
        # parameters.put('targetWindowSizeStr', '3x3') 
        # parameters.put('sigmaStr', '0.9') 
        # parameters.put('anSize', '50')
        speckle_filter = snappy.GPF.createProduct('Speckle-Filter', parameters, product)

        return speckle_filter

    def terrain_correction(self, product):
        """
        TODO: check 
        """
        parameters = snappy.HashMap() 
        parameters.put('demName', 'SRTM 3Sec')
        parameters.put('imgResamplingMethod', 'BILINEAR_INTERPOLATION')
        parameters.put('demResamplingMethod', 'BILINEAR_INTERPOLATION') 
        parameters.put('pixelSpacingInMeter', 10.0) 
        parameters.put('nodataValueAtSea', False)
        parameters.put('mapProjection', 'AUTO:42001')
        parameters.put('saveSelectedSourceBand', True)
        terrain_corrected = snappy.GPF.createProduct("Terrain-Correction", parameters, product)

        return terrain_corrected

if __name__=='__main__':
    """ Just for testing purposes: """
    process = Preprocess('/localhome/studenter/mikaellv/Project/data/unprocessed_downloads/')

    product = process.read_product('S1B_IW_GRDH_1SDV_20200627T053825_20200627T053850_022215_02A297_D3A3.zip')
    product_info = process.get_product_info(product)
    subset_product = process.create_subset(product,21000,0,1000,1500)
    subset_info = process.get_product_info(subset_product)
    orbit_corrected_product = process.apply_orbit_file(subset_product)
    therm_noise_removed_product = process.thermal_noise_removal(orbit_corrected_product)
    rm_calibrated_product = process.calibrate(therm_noise_removed_product)
    filtered_product = process.speckle_filter(rm_calibrated_product)
    terrain_corrected_product = process.terrain_correction(filtered_product)

    # Write GTiff to file:
    out_path = '/localhome/studenter/mikaellv/Project/data/processed_plots/Final_VH'
    snappy.ProductIO.writeProduct(terrain_corrected_product,out_path,'GeoTIFF')

    process.plot_band(terrain_corrected_product,'Sigma0_VH',0,0.04,save_path='/localhome/studenter/mikaellv/Project/data/processed_plots/Final_VH.png')
    
    