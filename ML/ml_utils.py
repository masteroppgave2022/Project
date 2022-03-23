import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
import pandas as pd
# import tensorflow_datasets as tfds
import keras as keras
import keras.layers as layers
import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()
# import tensorflow_hub as hub
import rasterio
import datetime
import os
import seaborn as sns
#import cv2
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import configparser
import math
from data.plot_data import plotPred
from data.plot_data import plotMaskedImage

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from skimage import exposure

import albumentations as A

class ML_utils():
    def __init__(self) -> None:
        self.parser_ml = configparser.ConfigParser()
        self.parser_ml.read('/localhome/studenter/mikaellv/Project/ML/ml_config.ini')

        self.model_name = self.parser_ml['model']['NAME']

        self.EPOCHS1=int(self.parser_ml['train']['EPOCHS1'])
        self.EPOCHS2=int(self.parser_ml['train']['EPOCHS2'])
        self.BATCH_SIZE=int(self.parser_ml['train']['BATCH_SIZE'])
        self.HEIGHT= int(self.parser_ml['train']['HEIGHT'])
        self.WIDTH= int(self.parser_ml['train']['WIDTH'])

        self.CLASSES = {
            1: 'Water',
            0: 'not_water',
        }

        # self.CLASSES = {
        #     1: 'Water',
        #     2: 'Trees',
        #     3: 'Grass',
        #     4: 'Flooded Vegetation',
        #     5: 'Crops',
        #     6: 'Scrub/Shrub',
        #     7: 'Built Area',
        #     8: 'Bare Ground',
        #     9: 'Snow/Ice'
        # }

        self.N_CLASSES=len(self.CLASSES)

    # def LoadImage(self, name, path):
    #     """ Return: (h,w,n)-np.arrays """
    #     # Images to np-arrays
    #     image_arr = rasterio.open(os.path.join(path+'/images/',name)).read()
    #     mask_arr = rasterio.open(os.path.join(path+'/masks/',name)).read()
    #     # Convert dimensions to standard (n,height,width) --> (height,width,n)
    #     image = np.rollaxis(image_arr,0,3)
    #     mask = np.rollaxis(mask_arr,0,3)

    #     return image, mask

    def LoadImage(self, file, image_path, mask_path):
        """ Return: (h,w,n)-np.arrays """
        # Images to np-arrays
        image_arr = rasterio.open(image_path+'/'+file).read()
        mask_arr = rasterio.open(mask_path+'/'+file).read()
        # Convert dimensions to standard (n,height,width) --> (height,width,n)
        image = np.rollaxis(image_arr,0,3)
        mask = np.rollaxis(mask_arr,0,3)
        # Histogram stretch and normalize:
        bands = image.shape[-1]
        rescaled_image = np.zeros(image.shape,dtype="float32")
        for b in range(bands):
            p2 = np.percentile(image[:,:,b],2)
            p98 = np.percentile(image[:,:,b],98)
            rescaled_band = exposure.rescale_intensity(image[:,:,b],in_range=(p2,p98),out_range=(0,1))
            rescaled_image[:,:,b] = rescaled_band
        image = rescaled_image
        # Check for NaN values
        for i in image:
            for e in i:
                for a in range(len(e)):
                    if math.isnan(e[a]): e[a]=255
        return image, mask

    def bin_image(self, mask):
        bins = np.array([pixel_val for pixel_val in self.CLASSES.keys()])
        new_mask = np.digitize(mask, bins)
        return new_mask

    def image_augmentation(self, image:np.array, mask:np.array):
        transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.ChannelShuffle(p=0.2),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5)
        ])
        transformed = transform(image=image, mask=mask)
        return transformed['image'], transformed['mask']

    def getSegmentationArr(self, image, classes):
        width=self.WIDTH
        height=self.HEIGHT
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
            #seg_img[:,:,2] += (segc*( colors[c][2] ))

        return(seg_img)

    def DataGenerator(self, path, mask_folder, train=False):
        batch_size=self.BATCH_SIZE
        classes=self.N_CLASSES
        files = [f for f in os.listdir(path) if not f.startswith('.')] #os.listdir(path+'/images')
        while True:
            for i in range(0, len(files), batch_size):
                batch_files = files[i : i+batch_size]
                imgs=[]
                segs=[]
                for file in batch_files:
                    if not file.startswith('.'):
                        image, mask = self.LoadImage(file, path, mask_folder)
                        if train: image, mask = self.image_augmentation(image,mask)
                        mask_binned = self.bin_image(mask)
                        labels = self.getSegmentationArr(mask_binned, classes)
                        imgs.append(image[:,:,0:3])  #uses the first 3 bands 

                        # im = np.array([image[:,:,0], image[:,:,1], image[:,:,0]/image[:,:,1]])
                        # ima = np.rollaxis(im,0,3)
                        # #print(ima)

                        # for i in ima:
                        #     for e in i:
                        #         for a in range(len(e)):
                        #             if math.isnan(e[a]): e[a]=-1 # changed from 0 to -1

                        # imgs.append(ima)  #uses the first 2 bands and the 0/1 as a third band

                        # segs.append(mask)
                        # imgs.append(image)
                        segs.append(labels)
                        # plotMaskedImage(image, mask)

                        # fig, axs = plt.subplots(1, 4, figsize=(25,25))
                        # plt.tight_layout()
                        # axs[0].imshow(10*np.log10(image[:, :, 0]), cmap='ocean') 
                        # axs[0].set_title('0')
                        # axs[2].imshow(10*np.log10(image[:, :, 2]), cmap='ocean') 
                        # axs[2].set_title('1')
                        # axs[1].imshow(10*np.log10(image[:, :, 1]), cmap='ocean') 
                        # axs[1].set_title('2')
                        # axs[3].imshow(10*np.log10(image[:, :, 3]), cmap='ocean') 
                        # axs[3].set_title('3')
                        # plt.show()

                        # plt.imshow(image[:, :, 0])
                        # plt.imshow(image[:, :, 1])
                        # plt.imshow(image[:, :, 2])
                        # plt.imshow(image[:, :, 3])
                        #plt.imshow(labels[:, :, 0])
                        #plt.imshow(labels[:, :, 1])
                        # plt.show()
                yield np.array(imgs), np.array(segs)

    def Unet(self):
        # N = 3
        # inp = layers.Input(shape=(None, None, N))
        # l1 = layers.Conv2D(3, (1, 1))(inp)
        base_model = sm.Unet('resnet50', classes=self.N_CLASSES, activation='softmax', encoder_weights='imagenet', input_shape=[256, 256, 3], encoder_freeze=True)
        # out = base_model(inp)
        # model = keras.models.Model(inp, out, name=base_model.name)
        #model = sm.Unet('resnet50', classes=self.N_CLASSES, activation='softmax', encoder_weights='imagenet', input_shape=[self.HEIGHT, self.WIDTH, 4], encoder_freeze=True)
        #tf.keras.utils.plot_model(model, show_shapes=True, to_file=self.model_name+'.png')
        return base_model

    def Unet2(self):
        N = 3
        inp = layers.Input(shape=(None, None, N))
        l1 = layers.Conv2D(3, (1, 1))(inp)
        base_model = sm.Unet('resnet50', classes=self.N_CLASSES, activation='softmax')
        out = base_model(l1)
        model = keras.models.Model(inp, out, name=base_model.name)
        #model = sm.Unet('resnet50', classes=self.N_CLASSES, activation='softmax', encoder_weights='imagenet', input_shape=[self.HEIGHT, self.WIDTH, 4], encoder_freeze=True)
        #tf.keras.utils.plot_model(model, show_shapes=True, to_file=self.model_name+'.png')
        return model

    def Unet3(self):
        N=3
        model = sm.Unet('resnet50', classes=self.N_CLASSES, activation='softmax', encoder_weights='imagenet', input_shape=[None, None, 3], encoder_freeze=True)
        return model

    def plot_history(self, history):
        """
        Must be adapted to the content of the history dataframe
        """
        history_frame1 = pd.DataFrame(history.history)
        history_frame1.loc[:, ['loss', 'val_loss']].plot()
        history_frame1.loc[:, ['categorical_crossentropy', 'val_categorical_crossentropy']].plot()
        history_frame1.loc[:, ['acc', 'val_acc']].plot()
        print(history_frame1)
        plt.show()

def ML_main(train_folder,valid_folder, mask_folder, mask_folder_val ):

    num_training_samples = len(os.listdir(train_folder))#len(os.listdir(train_folder+'/images'))
    num_valid_samples = len(os.listdir(train_folder))#len(os.listdir(valid_folder+'/images'))

    ml = ML_utils()

    train_gen = ml.DataGenerator(train_folder, mask_folder, train=True)
    val_gen = ml.DataGenerator(valid_folder, mask_folder_val, train=True)

    imgs, segs = next(train_gen)

    # plotMaskedImage(imgs[5], segs[5])

    model = ml.Unet()
    model.summary()

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['categorical_crossentropy', 'acc'],
    )

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    checkpoint = ModelCheckpoint('model.hdf5', monitor='val_acc', verbose=1, save_best_only=False, mode='max')

    TRAIN_STEPS = num_training_samples//ml.BATCH_SIZE+1
    VAL_STEPS = num_valid_samples//ml.BATCH_SIZE+1

    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=ml.EPOCHS1,
        steps_per_epoch=TRAIN_STEPS,
        callbacks=[tensorboard_callback, checkpoint], #checkpoint,
        #workers=0,
        verbose=1,
        shuffle=True,
        validation_steps=VAL_STEPS,
    )

    sm.utils.set_trainable(model, recompile=False)

    model.summary()

    model.compile(
        optimizer=Adam(learning_rate=0.000001),
        loss='categorical_crossentropy',
        metrics=['categorical_crossentropy', 'acc'],
    )

    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=ml.EPOCHS2,
        steps_per_epoch=TRAIN_STEPS,
        callbacks=[checkpoint, tensorboard_callback],
        #workers=0,
        verbose=1,
        shuffle=True,
        validation_steps=VAL_STEPS,
    )

    model.save('/localhome/studenter/mikaellv/Project/ML/models/' + ml.model_name)

    # ml.plot_history(history1)
    # ml.plot_history(history2)

    # max_show = 20
    # imgs, segs = next(val_gen)
    # pred = model.predict(imgs)

    # predictions = []
    # segmentations = []
    # for i in range(len(pred)):
    #     predictions.append(np.argmax(pred[i], axis=-1))
    #     segmentations.append(np.argmax(segs[i], axis=-1))

    # for i in range(max_show):
    #     plotPred(imgs[i], segs[i], predictions[i])

    # for i in range(max_show):
    #    plotPred(imgs[i], np.argmax(segs[i], axis=-1), np.argmax(pred[i], axis=-1))
    
