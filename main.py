import os
import logging
import configparser



#####################   ML
from ML.ml_utils import ML_utils 
from ML.ml_utils import ML_main
import tensorflow as tf
import keras as keras
import datetime
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()
from data.plot_data import plotMaskedImage
from data.plot_data import plotPred
import data.plot_data as pd

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

import numpy as np

import seaborn as sns

def give_color_to_seg_img(seg):
    n_classes=2
    seg_img = np.zeros( (seg.shape[0],seg.shape[1],3) ).astype('float')
    colors = sns.color_palette("hls", n_classes)

    for c in range(n_classes):
        segc = (seg == c)
        seg_img[:,:,0] += (segc*( colors[c][0] ))
        seg_img[:,:,1] += (segc*( colors[c][1] ))
        #seg_img[:,:,2] += (segc*( colors[c][2] ))

    return(seg_img)


""" Log progress """
logging.basicConfig(filename='main_log.log', encoding='utf-8', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s',  datefmt='%m/%d/%Y %H:%M:%S')

"""
Initiate parser for main_config.ini 

main_config.ini is the main setup file. 
To only run parts of the program, edit main_config.ini.
"""
parser_main = configparser.ConfigParser()
parser_main.read('/localhome/studenter/renatask/Project/main_config.ini')
root = parser_main['main']['root']
download_path = root + 'unprocessed_downloads/'
shapefile_path = parser_main['main']['shapefile_path']
subset_path = parser_main['main']['subset_path']
save_path = parser_main['main']['save_path']

if parser_main.getboolean('main','download'):
    """ Request and download Sentinel-1 images from from Alaska satellite facility """
    import data.request as req

    user = parser_main['download']['username']
    pw = parser_main['download']['password']
    search_config_path = root+'search_configs/' # Config_files for specific Sentinel-1 regions and acquisitions
    parser_locations = configparser.ConfigParser()
    parser_locations.read(search_config_path+'LOCATIONS.ini')
    configs = []
    for f in os.listdir(search_config_path):
        loc = f.split('.')[0]
        if loc.lower() == 'locations': continue
        if parser_locations.getboolean('download',loc):
            configs.append(search_config_path+f)
    request = req.requestDownload(username=user,password=pw,search_configs=configs)


if parser_main.getboolean('main', 'preprocess'):
    """ Image processing and dataset build """
    from data.preprocess_functions import Preprocess

    pp = Preprocess()
    if parser_main.getboolean('preprocess', 'clip_shapefile'):
        print('[INFO] Clipping label shapefiles ... ')
        src = root + 'FKB_vann/FKB_vann.shp'
        dest = root + 'untiled_masks/'
        extent_root = root + 'shapefiles/'
        locations = [loc for loc in os.listdir(extent_root) if not loc.startswith('.')]
        extents = [extent_root+f+'/'+f+'.shp' for f in locations]
        pp.clip_shapefile(src,extents,dest)
        print('[INFO] Done with all!')
    
    # Processing images:
    for file in os.listdir(download_path):
        if file.startswith("."): continue # Skip invalid system-generated hidden files
        product = pp.read_product(download_path + file)
        i = 0
        for shape in os.listdir(shapefile_path):
            name = shape+"_"+file[0:32]
            # Skip if invalid file or processed image is already generated
            if shape.startswith(".") or name+'.tif' in os.listdir(save_path):
                logging.info(f"[SKIPPING] File already generated or invalid: {name}.")
                continue
            # Processing pipeline:
            try:
                subset = pp.subset(product, shapefile_path+shape+"/"+shape)
            except Exception as e:
                logging.info(str(e) + f"\n\n[SKIPPING] Snappy could not read file: {file}.")
                continue
            if subset: # Check if subset is empty (None)
                logging.info(f"[INFO] Subset {shape+'_'+file} created")
                try:
                    subset_O = pp.apply_orbit_file(subset)
                    logging.info(f"[INFO] Orbitfile applied to {name}")
                except:
                    logging.info(f"[INFO] Orbit file failed/not found for {name}.")
                    break
                subset_O_TNR = pp.apply_thermal_noise_removal(subset_O)
                logging.info(f"[INFO] Thermal noise removal for {name} finished.")
                subset_O_TNR_C = pp.calibrate(subset_O_TNR)
                logging.info(f"[INFO] Calibration for {name} finished.")
                subset_O_TNR_C_S = pp.speckle_filter(subset_O_TNR_C)
                logging.info(f"[INFO] Speckle filter of {name} finished.")
                subset_O_TNR_C_S_TC = pp.terrain_correction(subset_O_TNR_C_S)
                logging.info(f"[INFO] Terrain correction for {name} finished.")
                subset_O_TNR_C_S_TC_DEM = pp.add_elevation_band(subset_O_TNR_C_S_TC)
                logging.info(f"[INFO] Elevation band added to {name}.")
                pp.save_product(subset_O_TNR_C_S_TC_DEM, name, save_path)
                pp.clip_raster(save_path+name+'.tif',shapefile_path+shape,save_path,name.split('.')[0])
                logging.info(f"[INFO] Subset {name} preprocessed and saved.")

if parser_main.getboolean('main','build_data'):
    """ Build dataset with processed images and masks """
    import data.build_data as bd

    image_root = root + 'processed_downloads/'
    tiled_images_root = root + 'tiled_images/'
    mask_root = root + 'processed_masks/'
    tiled_masks_root = root + 'tiled_masks/'

    # Convert SHP to GTIFF Files
    logging.info("[INFO] Converting shapefiles to GTiff rasters ...")
    mask_shp_paths = root + 'untiled_masks/'
    shp_paths = [mask_shp_paths+l+'/'+l+'.shp' for l in os.listdir(shapefile_path) if not l.startswith('.')]
    out_path = 'data/processed_masks/'
    for shp in shp_paths:
        bd.shp_to_gtiff(path_to_shp=shp,out_path=out_path)

    # Tile images
    if parser_main.getboolean('build_data','tile_images'):
        image_paths = [image_root+img for img in os.listdir(image_root) if not img.startswith('.')]
        for image in image_paths:
            name = os.path.split(image)[1].split('.')[0]
            out_path = tiled_images_root+name+'/'
            if os.path.exists(out_path): continue
            os.makedirs(out_path)
            bd.tile_write(image_path=image, output_path=out_path)

    # Tile masks
    if parser_main.getboolean('build_data','tile_masks'):
        mask_paths = [mask_root+m for m in os.listdir(mask_root) if not m.startswith('.')]
        for mask in mask_paths:
            name = os.path.split(mask)[1].split('.')[0]
            out_path = tiled_masks_root+name+'/'
            if os.path.exists(out_path): continue
            os.makedirs(out_path)
            bd.tile_write(image_path=mask, output_path=out_path)

    # Build dataset of image- and mask tiles
    dataset_name = parser_main['build_data']['dataset_name']
    train_val_split = float(parser_main['build_data']['train_val_split'])
    image_paths = [tiled_images_root + img + '/' for img in os.listdir(tiled_images_root) if not img.startswith('.')]
    order_of_regions = [region.split('/')[-2].split('_S1')[0] for region in image_paths]
    paths_to_tiles = []
    for image in image_paths:
        for tile in os.listdir(image):
            paths_to_tiles.append(image + tile)
    mask_dirs = []
    paths_to_mask_tiles = []
    for region in order_of_regions:
        dir = tiled_masks_root + region + '/'
        mask_dirs.append(dir)
    for directory in mask_dirs:
        for tile in os.listdir(directory):
            if not tile.startswith('.'): paths_to_mask_tiles.append(directory + tile)
    logging.info("[INFO] Building dataset ...")
    bd.build_dataset(destination_path = root + 'datasets/',\
        dataset_name = dataset_name,
        split = train_val_split,
        images = paths_to_tiles,
        masks = paths_to_mask_tiles)

if parser_main.getboolean('main','ML'):
    dataset = parser_main['ML']['dataset']
    train_images = 'data/datasets/' + dataset + '/train/images'
    train_masks = 'data/datasets/' + dataset + '/train/masks'
    val_images = 'data/datasets/' + dataset + '/val/images'
    val_masks = 'data/datasets/' + dataset + '/val/masks'

    if parser_main.getboolean('ML','train'):
        num_train_samples = len([img for img in os.listdir(train_images) if not img.startswith('.')]) # Exclude hidden files starting with '.'
        num_val_samples = len([img for img in os.listdir(train_masks) if not img.startswith('.')])

<<<<<<< Updated upstream
        ML_main(train_folder+'images', valid_folder+'images', mask_folder, mask_folder_val)
=======
        ML_main(train_images, val_images, train_masks, val_masks)
>>>>>>> Stashed changes
        
    if parser_main.getboolean('ML','val'):
        ml = ML_utils()

<<<<<<< Updated upstream
        # data = '/localhome/studenter/renatask/Project/data/tiled_images/melhus_lakes_S1A_IW_GRDH_1SDV_20200628T164709'
        # masks = '/localhome/studenter/renatask/Project/data/tiled_masks/melhus_lakes'
        # val_gen = ml.DataGenerator(data, masks)
        val_gen = ml.DataGenerator(valid_folder+'images', mask_folder_val)

        model = keras.models.load_model("/localhome/studenter/renatask/Project/ML/models/model_test")
=======
        val_gen = ml.DataGenerator(val_images, val_masks)

        model = keras.models.load_model("/localhome/studenter/mikaellv/Project/ML/models/test_model_2")
>>>>>>> Stashed changes
        model.summary()

        max_show = 20
        #imgs, segs =  ml.DataGenerator('/localhome/studenter/renatask/Project/data/tiled_images/gaula_melhus_S1A_IW_GRDH_1SDV_20200913T165517', '/localhome/studenter/renatask/Project/data/tiled_masks/gaula_melhus') #val_gen
        #pred = model.predict(imgs)
        imgs, segs = next(val_gen)
        pred = model.predict(imgs)


        predictions = []
        segmentations = []
        for i in range(0,len(pred),10):
            #p = give_color_to_seg_img(pred[i])
            p = np.argmax(pred[i], axis=-1)
            print(pred[i])
            print(p)
            predictions.append(np.argmax(pred[i], axis=-1))
            segmentations.append(np.argmax(segs[i], axis=-1))

        for i in range(max_show):
            plotPred(imgs[i], segs[i], p)
            
            
            

        print(f'preds: {predictions[1]}')
        print(f'segs: {segmentations[1]}')

        segmentations = np.array(segmentations)
        predictions = np.array(predictions)

        print(f"segmentation shape: {segmentations.shape}")
        print(f"predictions shape: {predictions.shape}")

        pred1D = predictions.reshape(-1)
        segs1D = segmentations.reshape(-1)

        print(f"segmentation 1d: {segs1D.shape}")
        print(f"predictions 1d: {pred1D.shape}")

        print(f"Confusion matrix: \n {tf.math.confusion_matrix(segs1D, pred1D, num_classes=ml.N_CLASSES)}")

        precision = precision_score(segs1D, pred1D, average='weighted')
        recall = recall_score(segs1D, pred1D, average='weighted')
        print(f'Precision score: {precision}')
        print(f'Recall score: {recall}')
        print(f'F1 score: {(2*precision*recall)/(recall+precision)}')
        print(f"Confusion matrix: \n {confusion_matrix(segs1D, pred1D)}")
        f1=f1_score(segs1D, pred1D, average='weighted')
        print(f'F1 score: {f1}')