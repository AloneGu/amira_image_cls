#!/usr/bin/env python
# encoding: utf-8

"""
@author: Jackling Gu
@file: apps.py
@time: 2017-06-12 17:04
"""

import flask
from .util import getenv

# init
app = flask.Flask(__name__)


# test app
@app.route('/hello')
def hello():
    return 'hello main page'


# function to set cfg
def setcfg(name, default=''):
    value = getenv(name, default=default)
    app.config[name] = value


setcfg('IMG_WIDTH', 224)
setcfg('IMG_HEIGHT', 224)
setcfg('MODEL_NAME', 'ALEXNET')
setcfg('EPOCH', 3)
setcfg('DATA_DIR', '../data/dog_vs_cat')

from .model.img_process_page import img_api

app.register_blueprint(img_api, url_prefix='/image')
