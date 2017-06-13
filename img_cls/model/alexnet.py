#!/usr/bin/env python
# encoding: utf-8

"""
@author: Jackling Gu
@file: alexnet.py
@time: 2017-06-12 17:05
"""

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization


# AlexNet with batch normalization in Keras
# input image is 224x224

class AlexNet(object):
    def __init__(self, h, w, num_class):
        self.h = h
        self.w = w
        self.num_class = num_class

    def get_model(self):
        model = Sequential()

        model.add(Conv2D(64, kernel_size=11, input_shape=(self.h, self.w, 3), strides=4, padding='valid'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

        model.add(Conv2D(192, kernel_size=5, padding='valid'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(384, kernel_size=3, padding='valid'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, kernel_size=3, padding='valid'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(4096))
        model.add(Activation('relu'))
        model.add(Dense(4096))
        model.add(Activation('relu'))
        model.add(Dense(self.num_class, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        return model
