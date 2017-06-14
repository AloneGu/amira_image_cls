#!/usr/bin/env python
# encoding: utf-8

"""
@author: Jackling Gu
@file: img_process.py
@time: 2017-06-13 11:13
"""

from ..util import getcfg, data_load, get_abspath
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from keras.callbacks import CSVLogger
from keras.utils import to_categorical
from keras.applications import vgg16, vgg19, inception_v3
from scipy.misc import imread, imresize
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator


class ImageClassification(object):
    def __init__(self):

        self.img_w = getcfg('IMG_WIDTH', 224)
        self.img_h = getcfg('IMG_HEIGHT', 224)
        self.epoch = getcfg('EPOCH', 10)
        self.default_shape = (3, self.img_h, self.img_w)  # channel first

        self.model_name = getcfg('MODEL_NAME', 'ALEXNET')
        self.data_path = getcfg('DATA_DIR', '../data/dog_vs_cat')
        self.model_save_path = get_abspath('../models/{}_{}_model.h5'.format(self.model_name, self.epoch))
        print('MODEL NAME', self.model_name, 'EPOCHS', self.epoch, 'DATA PATH', self.data_path)
        print('MODEL SAVE PATH', self.model_save_path)

        # get data
        self.x, self.y = data_load(self.data_path, img_height=self.img_h, img_width=self.img_w)
        print('x shape', self.x.shape)
        self.encoder = LabelEncoder()
        self.label_y = self.encoder.fit_transform(self.y)
        self.num_class = len(self.encoder.classes_)
        if self.num_class == 2:
            self.binary_y = self.label_y
        else:
            self.binary_y = to_categorical(self.label_y)
        print(self.num_class, self.y[:2], self.label_y[:2], self.binary_y[:2])

        if os.path.exists(self.model_save_path):
            print('LOAD EXIST MODEL')
            self.model = load_model(self.model_save_path)
        else:
            self.model = None
            if self.model_name == 'ALEXNET':
                from .alexnet import AlexNet
                self.model = AlexNet(self.img_h, self.img_w, self.num_class).get_model()
            if self.model_name == 'SIMPLENET':
                from .simple_cnn import SimpleNet
                self.model = SimpleNet(self.img_h, self.img_w, self.num_class).get_model()
            elif self.model_name == 'VGG16':
                self.model = vgg16.VGG16(input_shape=self.default_shape)
            elif self.model_name == 'VGG19':
                self.model = vgg19.VGG19(input_shape=self.default_shape)
            elif self.model_name == 'INCEPTIONV3':
                self.model = inception_v3.InceptionV3(input_shape=self.default_shape)
            if self.model is not None:
                self.model.summary()
                self.train()

    def train(self):
        # use data augmentation
        datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)  # randomly flip images
        # self.model.fit(self.x, self.binary_y, epochs=self.epoch, validation_split=0.2)
        log_path = get_abspath('../models/{}_{}_training.log'.format(self.model_name, self.epoch))
        csv_logger = CSVLogger(log_path)
        self.model.fit_generator(datagen.flow(self.x, self.binary_y), steps_per_epoch=32, epochs=self.epoch, verbose=1,
                                 callbacks=[csv_logger])
        self.model.save(self.model_save_path)

    def run(self, img_file_path):
        img = imread(img_file_path)
        img = imresize(img, (self.img_h, self.img_w))
        img = np.transpose(img, (2, 1, 0))
        np_img = np.array([img])
        if self.num_class == 2:
            class_type = int(self.model.predict(np_img)[0])  # binary class 0 or 1
            prob = round(self.model.predict_proba(np_img)[0], 2)
        else:
            res = self.model.predict_proba(np_img)[0]
            class_type = np.argmax(res)
            prob = round(max(res), 2)
        return_res = {'type': self.encoder.inverse_transform(class_type), 'probability': prob}
        return return_res

