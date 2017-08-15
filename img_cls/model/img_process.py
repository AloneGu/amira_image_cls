#!/usr/bin/env python
# encoding: utf-8

"""
@author: Jackling Gu
@file: img_process.py
@time: 2017-06-13 11:13
"""

from ..util import getcfg, data_load, get_abspath, get_y_labels
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model, Model
from keras.callbacks import CSVLogger
from keras.utils import to_categorical
from keras.applications import vgg16, vgg19, inception_v3
from scipy.misc import imread, imresize
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator


def update_app_model(tmp_model, num_class):
    from keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
    # add a global spatial average pooling layer
    x = tmp_model.output
    try:
        x = Flatten()(x)  # vgg ?
    except:
        pass  # inception

    x = Dense(256, activation='relu', name='fc1')(x)
    x = Dropout(0.3)(x)

    if num_class == 2:
        # prediction layer
        x = Dense(1, activation='sigmoid', name='final_predictions')(x)
        my_model = Model(inputs=tmp_model.input, outputs=x)
        my_model.compile(loss='binary_crossentropy',
                         optimizer='rmsprop',
                         metrics=['accuracy'])
    else:
        # prediction layer
        x = Dense(num_class, activation='softmax', name='final_predictions')(x)
        my_model = Model(inputs=tmp_model.input, outputs=x)
        my_model.compile(loss='categorical_crossentropy',
                         optimizer='rmsprop',
                         metrics=['accuracy'])

    return my_model


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

        # get y information first
        self.encoder = LabelEncoder()
        self.encoder.fit(get_y_labels(self.data_path))
        self.num_class = len(self.encoder.classes_)

        if os.path.exists(self.model_save_path):
            # load model , do not have to load x data
            print('LOAD EXIST MODEL')
            self.model = load_model(self.model_save_path)
        else:
            # get data
            self.x, self.y = data_load(self.data_path, img_height=self.img_h, img_width=self.img_w)
            print('x shape', self.x.shape)

            self.label_y = self.encoder.transform(self.y)
            if self.num_class == 2:
                self.binary_y = self.label_y
            else:
                self.binary_y = to_categorical(self.label_y)
            print(self.num_class, self.y[:2], self.label_y[:2], self.binary_y[:2])

            # already shuffle ,split
            tmp_data_cnt = len(self.x)
            self.train_data_cnt = int(tmp_data_cnt * 0.65)
            self.x_train = self.x[:self.train_data_cnt]
            self.x_test = self.x[self.train_data_cnt:]
            self.y_train = self.binary_y[:self.train_data_cnt]
            self.y_test = self.binary_y[self.train_data_cnt:]

            self.model = None
            if self.model_name == 'ALEXNET':
                from .alexnet import AlexNet
                self.model = AlexNet(self.img_h, self.img_w, self.num_class).get_model()
            if self.model_name == 'SIMPLENET':
                from .simple_cnn import SimpleNet
                self.model = SimpleNet(self.img_h, self.img_w, self.num_class).get_model()
                # remove top fully connection layers and do not use imagenet weights ( hard to download )
            elif self.model_name == 'VGG16':
                tmp_model = vgg16.VGG16(input_shape=self.default_shape, include_top=True, weights='imagenet')
                self.model = update_app_model(tmp_model, self.num_class)
            elif self.model_name == 'VGG19':
                tmp_model = vgg19.VGG19(input_shape=self.default_shape, include_top=True, weights='imagenet')
                self.model = update_app_model(tmp_model, self.num_class)
            elif self.model_name == 'INCEPTIONV3':
                tmp_model = inception_v3.InceptionV3(input_shape=self.default_shape, include_top=True,
                                                     weights='imagenet')
                self.model = update_app_model(tmp_model, self.num_class)
            elif self.model_name == 'DENSENET':
                from .densenet import DenseNet
                tmp_model = DenseNet((3, self.img_h, self.img_w), depth=10, growth_rate=3,
                                     nb_filter=4)  # change to small size
                self.model = update_app_model(tmp_model, self.num_class)
            if self.model is not None:
                self.model.summary()
                self.train()

    def train(self):
        # use data augmentation
        datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.15,
            rotation_range=0.15,
            zoom_range=0.15,
            vertical_flip=True,
            horizontal_flip=True)  # randomly flip images
        # self.model.fit(self.x, self.binary_y, epochs=self.epoch, validation_split=0.2)
        log_path = get_abspath('../models/{}_{}_training.log'.format(self.model_name, self.epoch))
        csv_logger = CSVLogger(log_path)
        bat_size = 50
        steps = int(self.train_data_cnt / bat_size) + 20
        self.model.fit_generator(datagen.flow(self.x_train, self.y_train, batch_size=bat_size),
                                 steps_per_epoch=steps,
                                 validation_data=(self.x_test, self.y_test),
                                 epochs=self.epoch, verbose=1,
                                 callbacks=[csv_logger])
        self.model.save(self.model_save_path)

    def run(self, img_file_path):
        img = imread(img_file_path)
        img = imresize(img, (self.img_h, self.img_w))
        img = np.transpose(img, (2, 1, 0))
        np_img = np.array([img])
        if self.num_class == 2:
            pred = self.model.predict(np_img)[0]
            class_type = int(pred)  # binary class 0 or 1
        else:
            pred = self.model.predict_proba(np_img)[0]
            class_type = np.argmax(pred)
        return_res = {'type': self.encoder.inverse_transform(class_type), 'prediction': str(pred)}
        return return_res
