"""
based on http://step.esa.int/docs/tutorials/Performing%20SAR%20processing%20in%20Python%20using%20snappy.pdf
"""

import numpy as np
import matplotlib
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

# For shapefiles
import shapefile 
import pygeoif

class Preprocess():
    def __init__(self, path_to_data="downloads") -> None:
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
        band = product.getBand(band) 
        w = band.getRasterWidth()
        h = band.getRasterHeight() 
        print(w, h)

        band_data = np.zeros(w * h, np.float32) 
        band.readPixels(0, 0, w, h, band_data)

        band_data.shape = h, w

        width = 12
        height = 12
        plt.figure(figsize=(width, height))
        imgplot = plt.imshow(band_data, cmap=plt.cm.binary, vmin=vmin, vmax=vmax)
        plt.imshow(band_data, cmap=plt.cm.binary, vmin=vmin, vmax=vmax)
        if plt.figimage != None:
            plt.savefig(figname)
        else:
            plt.show(imgplot)

        return imgplot

    def apply_orbit_file(self, product):
        parameters = HashMap() 
        GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()
        parameters.put('Apply-Orbit-File', True)
        parameters.put('orbitType', 'Sentinel Precise (Auto Download)') 
        parameters.put('polyDegree', '3') 
        parameters.put('continueOnFail', 'false')

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

    def calibrate(self, product):
        """
        TODO: change to correct band and polarisation
        """
        parameters = HashMap() 
        parameters.put('outputSigmaBand', True) 
        parameters.put('sourceBands', 'Intensity_VV') 
        parameters.put('selectedPolarisations', "VV") 
        parameters.put('outputImageScaleInDb', False)
        product_calibrated = GPF.createProduct("Calibration", parameters, product)

        return product_calibrated

    def speckle_filter(self, product):
        """
        TODO: check values
        """
        filterSizeY = '5' 
        filterSizeX = '5' 
        parameters = HashMap()
        parameters.put('sourceBands', 'Sigma0_VV') 
        parameters.put('filter', 'Lee') 
        parameters.put('filterSizeX', filterSizeX) 
        parameters.put('filterSizeY', filterSizeY) 
        parameters.put('dampingFactor', '2') 
        parameters.put('estimateENL', 'true') 
        parameters.put('enl', '1.0') 
        parameters.put('numLooksStr', '1') 
        parameters.put('targetWindowSizeStr', '3x3') 
        parameters.put('sigmaStr', '0.9') 
        parameters.put('anSize', '50')
        speckle_filter = snappy.GPF.createProduct('Speckle-Filter', parameters, product)

        return speckle_filter

    def terrain_correction(self, product):
        """
        TODO: check 
        """
        parameters = HashMap() 
        parameters.put('demResamplingMethod', 'BILINEAR_INTERPOLATION') 
        parameters.put('imgResamplingMethod', 'BILINEAR_INTERPOLATION')
        parameters.put('demName', 'GETASSE30') #ASTER 1Sec GDEM SRTM 3Sec
        parameters.put('pixelSpacingInMeter', 10.0) 
        parameters.put('sourceBands', 'Sigma0_VV')
        terrain_corrected = GPF.createProduct("Terrain-Correction", parameters, product)

        return terrain_corrected

if __name__=='__main__':
    """ Just for testing purposes: """
    prosess = Preprocess()
    
    product = prosess.read_product("/S1B_IW_GRDH_1SDV_20200910T060300_20200910T060325_023309_02C443_0BCF.zip")

    info = prosess.get_product_info(product)

    print(info)
    prosess.plotBand(product, "Intensity_VV", 0, 100000, "testimage.png")

    product = prosess.apply_orbit_file(product)
    prosess.plotBand(product, "Intensity_VV", 0, 100000, "testimage_orbit.png")
    
    product = prosess.calibrate(product)
    prosess.plotBand(product, "Sigma0_VV", 0, 1, "testimage_calibrate.png")

    product = prosess.speckle_filter(product)
    prosess.plotBand(product, "Sigma0_VV", 0, 1, "testimage_speckle.png")

    product = prosess.terrain_correction(product)
    prosess.plotBand(product, "Sigma0_VV", 0, 0.1, "testimage_terraincorrected.png")

    