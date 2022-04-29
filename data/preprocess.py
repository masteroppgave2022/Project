import matplotlib
# matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

from preprocess_functions import Preprocess

import logging

if __name__=='__main__':
    logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    prosess = Preprocess()

    logging.info("process started")
    product = prosess.read_product("/localhome/studenter/renatask/Project/data/unprocessed_downloads/S1B_IW_GRDH_1SDV_20200910T060300_20200910T060325_023309_02C443_0BCF.zip")

    info = prosess.get_product_info(product)

    print(info)

    logging.info(info)

    subset = prosess.add_shape_file(product,"/localhome/studenter/renatask/Project/data/shapefiles-old/molde/molde.shp")
    #prosess.plotBand(subset, "Intensity_VV", 0, 100000)

    ######### Orbitfile

    subset_O = prosess.apply_orbit_file(subset)
    logging.info("Orbitfile applied")
    #prosess.plotBand(subset_O, "Intensity_VV", 0, 100000)  

    ######### Thermal Noise Reduction

    subset_TNR = prosess.apply_thermal_noise_removal(subset_O)
    logging.info("Thermal noise reduction applied")

    plt.figure(figsize=(24, 12))
    plt.suptitle("Orbit and TNR")
    plt.subplot(121)
    prosess.plotBand(subset_TNR, "Intensity_VH", 16, 20000) 
    plt.title("Intensity_VH")
    plt.subplot(122)
    prosess.plotBand(subset_TNR, "Intensity_VV", 16, 100000) 
    plt.title("Intensity_VV")
    #plt.show()

    ######### Calibration

    subset_C = prosess.calibrate(subset_TNR)
    logging.info("Calibration applied")

    info = prosess.get_product_info(subset_C)
    print(info)

    plt.figure(figsize=(24, 12))
    plt.suptitle("Orbit, TNR and Calibration")
    plt.subplot(121)
    prosess.plotBand(subset_C, "Sigma0_VH", 0, 0.1)
    plt.title("Sigma0_VH")
    plt.subplot(122)
    prosess.plotBand(subset_C, "Sigma0_VV", 0, 0.5)
    plt.title("Sigma0_VV")
    #plt.show()

    ######### Terrain Correction

    logging.warning("Terrain correction will take some time.....")
    
    subset_TC = prosess.terrain_correction(subset_C)

    info = prosess.get_product_info(subset_TC)

    print(info)

    plt.figure(figsize=(24, 12))
    plt.suptitle("Orbit, TNR, Calibration and Terrain Correction")
    plt.subplot(121)
    prosess.plotBand(subset_TC, "Sigma0_VH", 0, 0.1)
    plt.title("Sigma0_VH")
    plt.subplot(122)
    prosess.plotBand(subset_TC, "Sigma0_VV", 0, 0.5)
    plt.title("Sigma0_VV")
    plt.show()
    
    logging.info("Finished")

    