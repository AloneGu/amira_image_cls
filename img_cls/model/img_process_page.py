#!/usr/bin/env python
# encoding: utf-8

"""
@author: Jackling Gu
@file: img_process_page.py
@time: 2017-06-12 17:05
"""

import os
from flask import Blueprint, request, abort
import json
import time
import base64
from .img_process import ImageClassification


class ImgWorker(object):
    kUploadPage = '''<!doctype html>
       <title>Upload new File</title>
       <h1>Upload new File</h1>
       <form action="" method=post enctype=multipart/form-data>
           <p><input type=file name=image>
           <input type=submit value=Upload>
       </form>
       '''
    worker = None
    savedir = '/tmp/'


img_api = Blueprint("image_api", __name__)


@img_api.route('/hello')
def hello():
    return "hello img page"


@img_api.before_app_first_request
def my_init():
    ImgWorker.worker = ImageClassification()


@img_api.route('/img_classification')
def detect_object():
    if request.method == 'GET':
        return ImgWorker.kUploadPage

    if 'image' not in request.files:
        abort(400)
    img_file = request.files['image']
    if img_file.filename == '':
        abort(400)

    if ImgWorker.worker is None:
        return "initializing"

    filename = '%s-%s.jpg' % (int(time.time()), base64.b32encode(os.urandom(5)))
    save_path = os.path.join(ImgWorker.savedir, filename)
    img_file.save(save_path)

    result = ImgWorker.worker.run(save_path)
    return json.dumps(result)
