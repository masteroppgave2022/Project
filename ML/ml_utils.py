import os
import rasterio
import datetime
import configparser
import numpy as np
import pandas as pd
import seaborn as sns
import albumentations as A
from skimage import exposure
import matplotlib.pyplot as plt

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import keras
from keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from tensorflow.python.keras.optimizer_v2 import adam
from tensorflow.keras.applications import xception
from keras.callbacks import ModelCheckpoint, CSVLogger

import ML.DeepLabV3Plus.deeplabv3plus as dl
import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()

class ML_utils():
    def __init__(self, user:str=None) -> None:
        self.user = user
        self.parser_ml = configparser.ConfigParser()
        self.parser_ml.read('/localhome/studenter/'+self.user+'/Project/ML/ml_config.ini')

        self.model_name_1 = self.parser_ml['model']['NAME1']
        self.model_name_2 = self.parser_ml['model']['NAME2']
        self.EPOCHS1=int(self.parser_ml['train']['EPOCHS1'])
        self.EPOCHS2=int(self.parser_ml['train']['EPOCHS2'])
        self.BATCH_SIZE=int(self.parser_ml['train']['BATCH_SIZE'])
        self.HEIGHT= int(self.parser_ml['train']['HEIGHT'])
        self.WIDTH= int(self.parser_ml['train']['WIDTH'])
        self.CLASSES = {1: 'water', 0: 'not_water'}
        self.class_weights = {'not_water': 1.0, 'water': 1.0}
        self.N_CLASSES=len(self.CLASSES)

    def LoadImage(self, file, image_path, mask_path, fetch_mask=True):
        """ Return: (h,w,n)-np.arrays """
        image_arr = rasterio.open(image_path+'/'+file).read()
        image = np.rollaxis(image_arr,0,3)
        # Histogram stretch and normalize:
        bands = image.shape[-1]
        rescaled_image = np.zeros(image.shape,dtype="float32")
        for b in range(bands):
            p2 = np.percentile(image[:,:,b],2)
            p98 = np.percentile(image[:,:,b],98)
            rescaled_band = exposure.rescale_intensity(image[:,:,b],in_range=(p2,p98),out_range=(0,1))
            rescaled_image[:,:,b] = rescaled_band
        image = rescaled_image
        # image = xception.preprocess_input(rescaled_image)
        if fetch_mask:
            mask_arr = rasterio.open(mask_path+'/'+f"{file.split('_S1')[0]}.tif").read()
            mask = np.rollaxis(mask_arr,0,3)
            return image, mask
        else: return image

    def bin_image(self, mask):
        bins = np.array([pixel_val for pixel_val in self.CLASSES.keys()])
        new_mask = np.digitize(mask, bins)
        return new_mask

    def image_augmentation(self, image:np.array, mask:np.array):
        transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.ChannelShuffle(p=0.4),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5)
        ])
        transformed = transform(image=image, mask=mask)
        return transformed['image'], transformed['mask']

    def getSegmentationArr(self, image, classes):
        height = image.shape[0]
        width = image.shape[1]
        seg_labels = np.zeros((height, width, classes))
        img = image[:, :, 0]
        for c in range(classes):
            seg_labels[:, :, c] = (img == c ).astype(int)
        return seg_labels

    def give_color_to_seg_img(self, seg):
        n_classes=self.N_CLASSES
        seg_img = np.zeros( (seg.shape[0],seg.shape[1],2) ).astype('float')
        colors = sns.color_palette("hls", n_classes)
        for c in range(n_classes):
            segc = (seg == c)
            seg_img[:,:,0] += (segc*( colors[c][0] ))
            seg_img[:,:,1] += (segc*( colors[c][1] ))
        return(seg_img)

    def DataGenerator(self, path, mask_folder, train=False, masks=True):
        batch_size=self.BATCH_SIZE
        classes=self.N_CLASSES
        files = [f for f in os.listdir(path) if f.endswith('.tif')]
        while True:
            for i in range(0, len(files), batch_size):
                batch_files = files[i : i+batch_size]
                imgs=[]
                segs=[]
                sample_weights=[]
                for file in batch_files:
                    if file.startswith('.'): continue
                    if masks: 
                        image, mask = self.LoadImage(file,path,mask_folder,fetch_mask=True)
                        mask_binned = self.bin_image(mask)
                        labels = self.getSegmentationArr(mask_binned, classes)
                        segs.append(labels)
                    else:
                        image = self.LoadImage(file,path,mask_folder,fetch_mask=False)
                    if train and masks: image, mask = self.image_augmentation(image,mask)
                    imgs.append(image[:,:,0:3])
                    #sample_weights.append(self.add_sample_weights(mask))
                if train and masks: yield np.array(imgs), np.array(segs)
                elif masks and not train: yield imgs, segs
                else: yield imgs

    def add_sample_weights(self, label):
        class_weights = tf.constant([self.class_weights['not_water'],self.class_weights['water']])
        class_weights = class_weights/tf.reduce_sum(class_weights)
        sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32))
        return sample_weights

    def Unet(self):
        model = sm.Unet(
            'resnet50',
            classes=self.N_CLASSES,
            activation='softmax',
            encoder_weights='imagenet',
            input_shape=[None, None, 3],
            encoder_freeze=True)
        return model

    def DeepLabV3plus(self):
        # model = dl_test.DeeplabV3Plus(image_size=256,num_classes=2)
        model = dl.DeepLabv3Plus(
            input_shape=(256, 256, 3),
            classes=self.N_CLASSES,
            backbone='xception',
            weights='pascal_voc',
            activation='softmax',
            )
        return model

    def plot_history(self, history, name=None):
        """
        Must be adapted to the content of the history dataframe
        """
        history_frame = pd.DataFrame(history.history)
        history_frame.to_csv(f'/localhome/studenter/renatask/Project/ML/saved_dataframes/{name}.csv')
        try:
            plt.switch_backend("Agg")
            fig, axs = plt.subplots(1,3,figsize=(25,25))
            history_frame.loc[:, ['loss', 'val_loss']].plot(ax=axs[0])
            history_frame.loc[:, ['categorical_crossentropy', 'val_categorical_crossentropy']].plot(ax=axs[1])
            history_frame.loc[:, ['acc', 'val_acc']].plot().getfigure(ax=axs[2])
            if name:
                plt.savefig('ML/saved_dataframes/'+name+'.png')
            else: plt.show()
        except Exception as e:
            print(f"[ERROR]: {e}")

class CustomLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.math.sigmoid(y_pred)
        numerator = 2 * tf.reduce_sum(y_true * y_pred)+1
        denominator = tf.reduce_sum(y_true + y_pred)+1

        return 1 - numerator / denominator

def ML_main(train_folder, valid_folder, mask_folder, mask_folder_val, user:str=None, model_architecture:str='unet', train_loss='dice'):
    ml = ML_utils(user=user)

    num_training_samples = len(os.listdir(train_folder))
    num_valid_samples = len(os.listdir(train_folder))
    train_gen = ml.DataGenerator(train_folder, mask_folder, train=True)
    val_gen = ml.DataGenerator(valid_folder, mask_folder_val, train=True)

    if model_architecture == 'unet':
        model = ml.Unet()
    elif model_architecture == 'deeplab':
        model = ml.DeepLabV3plus()
        for layer in model.layers[:358]:
            layer.trainable = False
    elif model_architecture == 'restart_training':
        model = keras.models.load_model('ML/checkpoints/model.hdf5', custom_objects={'call':[CustomLoss.call]})
        restart_from = 26
        ml.EPOCHS1 -= restart_from
    else:
        raise Exception("Please provide a valid model_architecture: 'unet', 'deeplab', 'restart_training'")
    model.summary()

    if train_loss == 'dice':
        dice_loss = CustomLoss()
        loss = [dice_loss.call]
    else:
        loss = 'categorical_crossentropy'

    model.compile(
        optimizer=adam.Adam(learning_rate=1e-4),
        loss=loss,
        metrics=['categorical_crossentropy', 'acc'],
    )
    

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint = ModelCheckpoint('ML/checkpoints/model.hdf5', monitor='val_acc', verbose=1, save_best_only=False, mode='max')
    csv_logger = CSVLogger('ML/csv_logs/'+ml.model_name_1+'_2.log')
    csv_logger = CSVLogger('ML/csv_logs/'+ml.model_name_1+'_2.log')
    TRAIN_STEPS = num_training_samples//ml.BATCH_SIZE+1
    VAL_STEPS = num_valid_samples//ml.BATCH_SIZE+1

    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=ml.EPOCHS1,
        steps_per_epoch=TRAIN_STEPS,
        callbacks=[checkpoint, csv_logger],
        verbose=1,
        shuffle=True,
        validation_steps=VAL_STEPS
    )

    model.save(f'/localhome/studenter/{user}/Project/ML/models/' + ml.model_name_1)
    
    sm.utils.set_trainable(model, recompile=False)
    csv_logger = CSVLogger('ML/csv_logs/'+ml.model_name_1+'_3.log')
    model.summary()

    model.compile(
        optimizer=adam.Adam(learning_rate=1e-5),
        loss=loss,
        metrics=['categorical_crossentropy', 'acc'],
    )

    csv_logger = CSVLogger('ML/csv_logs/'+ml.model_name_2+'_3.log')

    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=ml.EPOCHS2,
        steps_per_epoch=TRAIN_STEPS,
        callbacks=[checkpoint, csv_logger],
        verbose=1,
        shuffle=True,
        validation_steps=VAL_STEPS
    )

    model.save('/localhome/studenter/'+user+'/Project/ML/models/' + ml.model_name_2)

    name1 = ml.model_name_2 + '_1.csv'
    name2 = ml.model_name_2 + '_2.csv'
    history_frame1 = pd.DataFrame(history1.history)
    history_frame1.to_csv(f'/localhome/studenter/{user}/Project/ML/saved_dataframes/{name1}.csv')
    history_frame2 = pd.DataFrame(history2.history)
    history_frame2.to_csv(f'/localhome/studenter/{user}/Project/ML/saved_dataframes/{name2}.csv')
