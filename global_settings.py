#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
# ==========================================
# Copyright 2024 Yang 
# PythonProjects - global
# ==========================================
#
#
# 
"""

import datetime
import os

ROOT = os.getcwd()

CAMERA_DIR = os.path.join(ROOT, r"statics\OpenCV Capture")
SAVE_DIR = os.path.join(ROOT, r"statics\Saved Pictures")
TEMPLATE_DIR = os.path.join(ROOT, r"statics\Template")

TEMPLATE_RESULTS_PATH = os.path.join(ROOT, r"template.txt")
TXT_PATH = os.path.join(ROOT, r"inside_temperature.txt")
EXAMPLE_IMG_PATH = os.path.join(ROOT, r"statics\examples\WIN_20240102_20_47_26_Pro.jpg")


def get_save_dir():
    now_time = datetime.datetime.now()
    save_dir = os.path.join(SAVE_DIR, datetime.datetime.strftime(now_time, "%Y-%m-%d"))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    return save_dir
