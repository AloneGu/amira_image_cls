#!/usr/bin/env python
# encoding: utf-8

"""
@author: Jackling Gu
@file: run.py
@time: 2017-06-13 11:03
"""

if __name__ == '__main__':
    from img_cls.apps import app

    app.run(host='0.0.0.0', port=8090)
