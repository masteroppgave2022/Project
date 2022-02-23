import os
import logging
import subprocess
import configparser
import multiprocessing
import snappy
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
#import jpy
from turtle import down
import data.request as req
from data.preprocess_functions import Preprocess 


def geopos_to_wkt(geopos):
    lat = []
    long =[]

    for e in geopos:
        lat.append(e.lat)
        long.append(e.lon)
    
    polygon_geom = Polygon(zip(long, lat))
    #print(polygon_geom)
    crs = {'init': 'epsg:4326'}
    polygon = gpd.GeoDataFrame(crs=crs, geometry=[polygon_geom])       
    #print(polygon.geometry)
    #geometry = gpd.points_from_xy(long, lat, crs="EPSG:4326")
    #wkt = geometry.GeoSeries.to_wkt()
    #polygon.to_file(filename='polygon.shp', driver="ESRI Shapefile")
    return polygon




if __name__ == '__main__':
    logging.basicConfig(filename='main_log.log', encoding='utf-8', level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s',  datefmt='%m/%d/%Y %H:%M:%S')
    parser_main = configparser.ConfigParser()
    parser_main.read('/localhome/studenter/mikaellv/Project/main_config.ini')
    root = parser_main['main']['root']
    download_path = parser_main['main']['download_path']
    shapefile_path = parser_main['main']['shapefile_path']
    subset_path = parser_main['main']['subset_path']
    save_path = parser_main['main']['save_path']

    if parser_main.getboolean('main','download'): # Download True/False
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
            if file.startswith("S1A_IW_GRDH_1SDV"): continue
            if file.startswith("S1A"): continue
            if file.startswith("S1B_IW_GRDH_1SDV_20200627T053825_"): continue
            product = pp.read_product(download_path+file)

            GeoPos = snappy.ProductUtils.createGeoBoundary(product, 1)
            one = GeoPos[0]
            print(GeoPos[2000])
            polygon = geopos_to_wkt(GeoPos)
            print(polygon)

            
            logging.info(f"Product {file} read")
            for shape in os.listdir(shapefile_path):
                if shape.startswith("."): continue
                #subset = pp.add_shape_file(product, shapefile_path+shape+"/"+shape)
                #pp.save_product(subset, shape+"_"+file, subset_path, "BEAM-DIMAP")
                subset = pp.subset(product, shapefile_path+shape+"/"+shape, shape+"_"+file, subset_path, GeoPos, "BEAM-DIMAP")
                if subset:
                    # logging.info(f"Subset {shape+'_'+file} saved")

        # for subset in os.listdir(subset_path):
                # if subset.endswith(".data"): continue
                # if subset.startswith("."): continue
                # logging.info(f"Subset {subset} read")
                # subset_R = pp.read_product(subset_path+subset)
                    subset_O = pp.apply_orbit_file(product=subset)
                    logging.info(f"Orbitfile applied to {subset}")
                    subset_O_TNR = pp.apply_thermal_noise_removal(subset_O)
                    logging.info(f"Thermal noise removal for {subset} finished")
                    subset_O_TNR_C = pp.calibrate(subset_O_TNR)
                    logging.info(f"Calibration for {subset} finished")
                    subset_O_TNR_C_TC = pp.terrain_correction(subset_O_TNR_C)
                    logging.info(f"Terrain correction for {subset} finished")
                    pp.save_product(subset_O_TNR_C_TC, 'test', save_path)
                    logging.info(f"Subset {subset} preprocessed and saved")
                    print("Finished.")



         
        pass




