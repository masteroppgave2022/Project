import matplotlib.pyplot as plt
import numpy as np
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

class ML_utils():
    def __init__(self) -> None:
        self.parser_ml = configparser.ConfigParser()
        self.parser_ml.read('/localhome/studenter/renatask/Project/ML/ml_config.ini')

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

        for i in image:
            for e in i:
                for a in range(len(e)):
                    if math.isnan(e[a]): e[a]=0

        return image, mask



    def bin_image(self, mask):
        bins = np.array([pixel_val for pixel_val in self.CLASSES.keys()])
        new_mask = np.digitize(mask, bins)
        return new_mask

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
        seg_img = np.zeros( (seg.shape[0],seg.shape[1],3) ).astype('float')
        colors = sns.color_palette("hls", n_classes)

        for c in range(n_classes):
            segc = (seg == c)
            seg_img[:,:,0] += (segc*( colors[c][0] ))
            seg_img[:,:,1] += (segc*( colors[c][1] ))
            seg_img[:,:,2] += (segc*( colors[c][2] ))

        return(seg_img)

    def DataGenerator(self, path, mask_folder):
        batch_size=self.BATCH_SIZE
        classes=self.N_CLASSES
        files = os.listdir(path)#os.listdir(path+'/images')
        while True:
            for i in range(0, len(files), batch_size):
                batch_files = files[i : i+batch_size]
                imgs=[]
                segs=[]
                for file in batch_files:
                    if not file.startswith('.'):
                        image, mask = self.LoadImage(file, path, mask_folder)
                        mask_binned = self.bin_image(mask)
                        labels = self.getSegmentationArr(mask_binned, classes)
                        imgs.append(image)
                        #segs.append(mask)
                        # imgs.append(image)
                        segs.append(labels)
                        #plt.imshow(image[:, :, 1])
                        #plt.imshow(labels[:, :, 0])
                        #plt.imshow(labels[:, :, 1])
                        #plt.show()
                yield np.array(imgs), np.array(segs)

    def Unet(self):
        N = 4
        inp = layers.Input(shape=(None, None, N))
        l1 = layers.Conv2D(3, (1, 1))(inp)
        base_model = sm.Unet('resnet50', classes=self.N_CLASSES, activation='softmax', encoder_weights='imagenet',encoder_freeze=True)
        out = base_model(l1)
        model = keras.models.Model(inp, out, name=base_model.name)
        #model = sm.Unet('resnet50', classes=self.N_CLASSES, activation='softmax', encoder_weights='imagenet', input_shape=[self.HEIGHT, self.WIDTH, 4], encoder_freeze=True)
        #tf.keras.utils.plot_model(model, show_shapes=True, to_file=self.model_name+'.png')
        return model

    def Unet2(self):
        N = 4
        inp = layers.Input(shape=(None, None, N))
        l1 = layers.Conv2D(3, (1, 1))(inp)
        base_model = sm.Unet('resnet50', classes=self.N_CLASSES, activation='softmax')
        out = base_model(l1)
        model = keras.models.Model(inp, out, name=base_model.name)
        #model = sm.Unet('resnet50', classes=self.N_CLASSES, activation='softmax', encoder_weights='imagenet', input_shape=[self.HEIGHT, self.WIDTH, 4], encoder_freeze=True)
        #tf.keras.utils.plot_model(model, show_shapes=True, to_file=self.model_name+'.png')
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

    # train_folder = parser_main['ML']['train_path']
    # valid_folder = parser_main['ML']['val_path']

    num_training_samples = len(os.listdir(train_folder))#len(os.listdir(train_folder+'/images'))
    num_valid_samples = len(os.listdir(train_folder))#len(os.listdir(valid_folder+'/images'))

    ml = ML_utils()

    train_gen = ml.DataGenerator(train_folder, mask_folder)
    val_gen = ml.DataGenerator(valid_folder, mask_folder_val)

    imgs, segs = next(train_gen)

    plotMaskedImage(imgs[1], segs[1])

    model = ml.Unet2()
    model.summary()

    model.compile(
        optimizer=Adam(),
        loss='categorical_crossentropy',
        metrics=['categorical_crossentropy', 'acc'],
    )

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    checkpoint = ModelCheckpoint('model.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    TRAIN_STEPS = num_training_samples//ml.BATCH_SIZE+1
    VAL_STEPS = num_valid_samples//ml.BATCH_SIZE+1

    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=ml.EPOCHS1,
        steps_per_epoch=TRAIN_STEPS,
        callbacks=[tensorboard_callback], #checkpoint,
        workers=0,
        verbose=1,
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
        workers=0,
        verbose=1,
        validation_steps=VAL_STEPS,
    )

    model.save(ml.model_name)

    ml.plot_history(history1)
    ml.plot_history(history2)

    max_show = 10
    imgs, segs = next(val_gen)
    pred = model.predict(imgs)

    for i in range(max_show):
        plotPred(imgs[i], np.argmax(segs[i], axis=-1), np.argmax(pred[i], axis=-1))
    
