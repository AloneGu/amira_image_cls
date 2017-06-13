#!/usr/bin/env python
# encoding: utf-8

"""
@author: Jackling Gu
@file: util.py
@time: 2017-06-13 09:30
"""

import os
import ast
import glob
from scipy.misc import imread, imresize
import numpy as np
from sklearn.utils import shuffle

# function to get os
def getcfg(name, default, app_=None):
    try:
        if app_ is None:
            from .apps import app
            app_ = app
        return app_.config[name]
    except:
        return default


def getenv(name, default=''):
    if name in os.environ:
        try:
            value = ast.literal_eval(os.environ[name])
        except (SyntaxError, ValueError):
            value = os.environ[name]
    else:
        value = default
    return value


def get_abspath(filename):
    return os.path.normpath(os.path.join(__file__, os.path.pardir, filename))


def data_load(data_dir_path, img_height, img_width):
    """

    :param data_dir_path: data home dir
    :return: x, y    x is image content, y is category name
    """
    data_dir = get_abspath(data_dir_path)
    subdirs = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
    x, y = [], []
    for cls in subdirs:
        imgs = glob.glob(os.path.join(cls, '*'))
        tmp_y = os.path.split(cls)[-1]
        for img_path in imgs:
            x.append(imresize(imread(img_path), (img_height, img_width)))
            y.append(tmp_y)
    return shuffle(np.array(x), y)
