#!/usr/bin/env python
# encoding: utf-8

"""
@author: Jackling Gu
@file: simple_cnn.py
@time: 2017-06-13 16:49
"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


class SimpleNet(object):
    def __init__(self, h, w, num_class):
        self.h = h
        self.w = w
        self.num_class = num_class

    def get_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=3, input_shape=(3, self.h, self.w)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, kernel_size=3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, kernel_size=3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        if self.num_class == 2:
            model.add(Dense(1))
            model.add(Activation('sigmoid'))
            model.compile(loss='binary_crossentropy',
                          optimizer='rmsprop',
                          metrics=['accuracy'])
            return model
        else:
            model.add(Dense(self.num_class, activation='softmax'))
            model.compile(loss='categorical_crossentropy',
                          optimizer='rmsprop',
                          metrics=['accuracy'])
            return model
