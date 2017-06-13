#!/usr/bin/env python
# encoding: utf-8

"""
@author: Jackling Gu
@file: img_process.py
@time: 2017-06-13 11:13
"""

from ..util import getcfg, data_load, get_abspath
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import load_model
from keras.layers import Input
from keras.applications import vgg16, vgg19, inception_v3
from scipy.misc import imread, imresize
import numpy as np
import os


class ImageClassification(object):
    def __init__(self):

        self.img_w = getcfg('IMG_WIDTH', 224)
        self.img_h = getcfg('IMG_HEIGHT', 224)
        self.epoch = getcfg('EPOCH', 10)
        self.default_shape = (self.img_h, self.img_w, 3)

        self.model_name = getcfg('MODEL_NAME', 'ALEXNET')
        self.data_path = getcfg('DATA_DIR', '../data/dog_vs_cat')
        self.model_save_path = get_abspath('../../models/{}_model.h5'.format(self.model_name))
        if os.path.exists(self.model_save_path):
            self.model = load_model(self.model_save_path)
        else:
            self.x, self.y = data_load(self.data_path, img_height=self.img_h, img_width=self.img_w)
            self.encoder = LabelEncoder()
            self.label_y = self.encoder.fit_transform(self.y)
            self.num_class = self.encoder.classes_[0]
            self.binary_y = to_categorical(self.label_y)
            self.model = None

            if self.model_name == 'ALEXNET':
                from .alexnet import AlexNet
                self.model = AlexNet(self.img_h, self.img_w, self.num_class).get_model()
            elif self.model_name == 'VGG16':
                self.model = vgg16.VGG16(input_shape=self.default_shape)
            elif self.model_name == 'VGG19':
                self.model = vgg19.VGG19(input_shape=self.default_shape)
            elif self.model_name == 'INCEPTIONV3':
                self.model = inception_v3.InceptionV3(input_shape=self.default_shape)
            if self.model is not None:
                self.train()

    def train(self):
        self.model.fit(self.x, self.binary_y, epochs=self.epoch, validation_split=0.2)
        self.model.save(self.model_save_path)

    def run(self, img_file_path):
        img = imread(img_file_path)
        img = imresize(img, (self.img_h, self.img_w))
        res = self.model.predict_proba(np.array([img]))[0]
        class_type = np.argmax(res)
        prob = max(res)
        return self.encoder.inverse_transform(class_type), prob
