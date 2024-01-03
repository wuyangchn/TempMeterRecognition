#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
# ==========================================
# Copyright 2024 Yang 
# PythonProjects - cv_funcs
# ==========================================
#
#
# 
"""

import cv2 as cv
import numpy as np


def cv_show(win_name, show_img):
    """
    :param win_name:
    :param show_img:
    :return:
    """
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(win_name, *show_img.shape[:2])
    cv.imshow(win_name, show_img)


def cv_resize(src_img, width=None, height=None, inter=cv.INTER_AREA):
    """
    调整图像尺寸
    :param src_img: 原图
    :param width: 调整后的宽
    :param height: 调整后的高
    :param inter: 插值方法
    :return: 调整后的图像
    """
    if width is None and height is None:
        return src_img  # 不予变换

    h, w = src_img.shape[:2]

    if width is None:
        rate = float(h) / height
        return cv.resize(src_img, (int(w / rate + 0.5), height), interpolation=inter)
    elif height is None:
        rate = float(w) / width
        return cv.resize(src_img, (width, int(h / rate + 0.5)), interpolation=inter)
    else:
        return cv.resize(src_img, (width, height), interpolation=inter)


def bounding_rect(contour: np.ndarray):
    c = contour.transpose()
    x = c[0][0].min()
    w = c[0][0].max() - x
    y = c[1][0].min()
    h = c[1][0].max() - y
    return int(x), int(y), int(w), int(h)


def write(file_path, params):
    """
    Parameters
    ----------
    file_path
    params

    Returns
    -------

    """
    with open(file_path, 'a+') as f:  # save serialized json data to a readable text
        f.writelines(params)
    return file_path