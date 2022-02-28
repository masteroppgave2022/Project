import os
import logging
import subprocess
import configparser
import multiprocessing
import snappy
from snappy import ProductIO
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
#import jpy
import data.build_data as bd
from turtle import down
import data.request as req
from data.preprocess_functions import Preprocess 

if __name__ == '__main__':
    logging.basicConfig(filename='main_log.log', encoding='utf-8', level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s',  datefmt='%m/%d/%Y %H:%M:%S')
    parser_main = configparser.ConfigParser()
    parser_main.read('/localhome/studenter/mikaellv/Project/main_config.ini')
    root = parser_main['main']['root']
    download_path = parser_main['main']['download_path']
    shapefile_path = parser_main['main']['shapefile_path']
    subset_path = parser_main['main']['subset_path']
    save_path = parser_main['main']['save_path']

    if parser_main.getboolean('main','download'):
        user = parser_main['download']['username']
        pw = parser_main['download']['password']
        search_configs = root+'search_configs/'
        parser_locations = configparser.ConfigParser()
        parser_locations.read(search_configs+'LOCATIONS.ini')
        configs = []
        for f in os.listdir(search_configs):
            loc = f.split('.')[0]
            if loc.lower() == 'locations': continue
            if parser_locations.getboolean('download',loc):
                configs.append(search_configs+f)
        request = req.requestDownload(username=user,password=pw,search_configs=configs)
    

    if parser_main.getboolean('main', 'preprocess'):
        """ Processing pipeline to be implemented here """
        print("[INFO] Processing...")
        pp = Preprocess()

        if parser_main.getboolean('preprocess', 'clip_shapefile'):
            print('[INFO] Clipping label shapefiles ... ')
            src = root + 'FKB_vann/FKB_vann.shp'
            dest = root + 'untiled_masks/'
            extent_root = shapefile_path
            locations = os.listdir(extent_root)
            locations.remove('.DS_Store')
            extents = [extent_root+f+'/'+f+'.shp' for f in locations]
            pp.clip_shapefile(src,extents,dest)
            print('[INFO] Done with all!')
        
        for file in os.listdir(download_path):
            if file.startswith("."): continue
            product = pp.read_product(download_path+file)
            try:
                GeoPos = snappy.ProductUtils.createGeoBoundary(product, 1)
                logging.info(f"Product {file} read")
            except:
                logging.info(f"Java NullPointerException, skipping...")
                continue
            for shape in os.listdir(shapefile_path):
                if shape.startswith("."): continue
                if shape+"_"+file+".tif" in os.listdir(save_path): continue
                #subset = pp.add_shape_file(product, shapefile_path+shape+"/"+shape)
                #pp.save_product(subset, shape+"_"+file, subset_path, "BEAM-DIMAP")
                name = shape+"_"+file[0:32]
                subset = pp.subset(product, shapefile_path+shape+"/"+shape, shape+"_"+file, subset_path, GeoPos, "BEAM-DIMAP")
                if subset: logging.info(f"Subset {shape+'_'+file} created")

                if subset:
                    try:
                        subset_O = pp.apply_orbit_file(product=subset)
                        logging.info(f"Orbitfile applied to {name}")
                    except:
                        logging.info(f"Orbit file failed/not found for {name}, skipping.")
                        continue
                    subset_O_TNR = pp.apply_thermal_noise_removal(subset_O)
                    logging.info(f"Thermal noise removal for {name} finished")
                    subset_O_TNR_C = pp.calibrate(subset_O_TNR)
                    logging.info(f"Calibration for {name} finished")
                    subset_O_TNR_C_TC = pp.terrain_correction(subset_O_TNR_C)
                    logging.info(f"Terrain correction for {name} finished")
                    # pp.plotBand(subset_O_TNR_C_TC, "Sigma0_VH", 0, 0.1)
                    pp.save_product(subset_O_TNR_C_TC, name, save_path)
                    pp.clip_raster(save_path+name+'.tif',shapefile_path+shape,save_path,name.split('.')[0])
                    logging.info(f"Subset {name} preprocessed and saved")
    
    if parser_main.getboolean('main','build_data'):
        """ Convert SHP to GTIFF Files """
        logging.info("Converting shapefiles to GTiff rasters ...")
        mask_shp_paths = root + 'untiled_masks/'
        shp_paths = [mask_shp_paths+l+'/'+l+'.shp' for l in os.listdir(shapefile_path) if not l.startswith('.')]
        out_path = 'data/processed_masks/'
        for shp in shp_paths:
            bd.shp_to_gtiff(path_to_shp=shp,out_path=out_path)

        if parser_main.getboolean('build_data','tile_images'):
            image_root = root+'processed_downloads/'
            tiled_root = root+'tiled_images/'
            image_paths = [image_root+img for img in os.listdir(image_root) if not img.startswith('.')]
            for image in image_paths:
                name = os.path.split(image)[1].split('.')[0]
                out_path = tiled_root+name+'/'
                if os.path.exists(out_path): continue
                os.makedirs(out_path)
                bd.tile_write(image_path=image, output_path=out_path)

        if parser_main.getboolean('build_data','tile_masks'):
            mask_root = root+'processed_masks/'
            tiled_root = root+'tiled_masks/'
            mask_paths = [mask_root+m for m in os.listdir(mask_root) if not m.startswith('.')]
            for mask in mask_paths:
                name = os.path.split(mask)[1].split('.')[0]
                out_path = tiled_root+name+'/'
                if os.path.exists(out_path): continue
                os.makedirs(out_path)
                bd.tile_write(image_path=mask, output_path=out_path)
                    




