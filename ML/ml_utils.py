import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()
import rasterio
import datetime
import os
import seaborn as sns
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import configparser
import math
from ML.deeplabV3plus.model import Deeplabv3
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
        self.CLASSES = {1: 'water', 0: 'not_water'}
        self.class_weights = {'not_water': 1.0, 'water': 5.0}
        self.N_CLASSES=len(self.CLASSES)

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

    def DataGenerator(self, path, mask_folder, train=False):
        batch_size=self.BATCH_SIZE
        classes=self.N_CLASSES
        files = [f for f in os.listdir(path) if not f.startswith('.')]
        while True:
            for i in range(0, len(files), batch_size):
                batch_files = files[i : i+batch_size]
                imgs=[]
                segs=[]
                sample_weights=[]
                for file in batch_files:
                    if file.startswith('.'): continue
                    image, mask = self.LoadImage(file, path, mask_folder)
                    if train: image, mask = self.image_augmentation(image,mask)
                    mask_binned = self.bin_image(mask)
                    labels = self.getSegmentationArr(mask_binned, classes)
                    imgs.append(image[:,:,0:3])
                    segs.append(labels)
                    sample_weights.append(self.add_sample_weights(mask))
                if train: yield np.array(imgs), np.array(segs)
                else: yield imgs, segs

    def Unet(self):
        model = sm.Unet('resnet50', classes=self.N_CLASSES, activation='softmax', encoder_weights='imagenet', input_shape=[None, None, 3], encoder_freeze=True)
        return model

    def DeepLabV3plus(self):
        model = Deeplabv3(
            input_shape=(None, None, 3),
            classes=self.N_CLASSES,
            backbone='xception',
            weights='cityscapes',
            activation='softmax',
            )
        return model

    def plot_history(self, history, name=None):
        """
        Must be adapted to the content of the history dataframe
        """
        history_frame = pd.DataFrame(history.history)
        history_frame.to_csv(f'/localhome/studenter/mikaellv/Project/ML/saved_dataframes/{name}.csv')
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

    def add_sample_weights(self, label):
        class_weights = tf.constant([self.class_weights['not_water'],self.class_weights['water']])
        class_weights = class_weights/tf.reduce_sum(class_weights)
        sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32))
        return sample_weights

def ML_main(train_folder,valid_folder, mask_folder, mask_folder_val ):

    num_training_samples = len(os.listdir(train_folder))#len(os.listdir(train_folder+'/images'))
    num_valid_samples = len(os.listdir(train_folder))#len(os.listdir(valid_folder+'/images'))

    ml = ML_utils()

    train_gen = ml.DataGenerator(train_folder, mask_folder, train=True)
    val_gen = ml.DataGenerator(valid_folder, mask_folder_val, train=True)

    model = ml.DeepLabV3plus()
    model.summary()

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['categorical_crossentropy', 'acc'],
    )

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    checkpoint = ModelCheckpoint('ML/checkpoints/model.hdf5', monitor='val_acc', verbose=1, save_best_only=False, mode='max')

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
        validation_steps=VAL_STEPS
    )

    # sm.utils.set_trainable(model, recompile=False)

    # model.summary()

    # model.compile(
    #     optimizer=Adam(learning_rate=0.000001),
    #     loss='categorical_crossentropy',
    #     metrics=['categorical_crossentropy', 'acc'],
    # )

    # history2 = model.fit(
    #     train_gen,
    #     validation_data=val_gen,
    #     epochs=ml.EPOCHS2,
    #     steps_per_epoch=TRAIN_STEPS,
    #     callbacks=[checkpoint, tensorboard_callback],
    #     #workers=0,
    #     verbose=1,
    #     shuffle=True,
    #     validation_steps=VAL_STEPS
    # )

    model.save('/localhome/studenter/mikaellv/Project/ML/models/' + ml.model_name)

    ml.plot_history(history1, name=ml.model_name+'_1.csv')
    # ml.plot_history(history2, name=ml.model_name+'_2.csv')

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
    
