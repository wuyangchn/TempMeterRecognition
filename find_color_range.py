#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
# ==========================================
# Copyright 2024 Yang 
# PythonProjects - find_color_range
# ==========================================
#
#
# 
"""
from global_settings import EXAMPLE_IMG_PATH
from basic_funcs import cv_resize, cv_show, cv, np


def find_color_range(example_img=""):
    if example_img == "":
        example_img = EXAMPLE_IMG_PATH

    Winname = "Frame:"

    def nothing(x):
        pass

    cv.namedWindow('Frame:')
    # H, S,V are for Lower Boundaries
    # H2,S2,V2 are for Upper Boundaries
    cv.createTrackbar('H', Winname, 0, 255, nothing)
    cv.createTrackbar('S', Winname, 0, 255, nothing)
    cv.createTrackbar('V', Winname, 0, 255, nothing)
    cv.createTrackbar('H2', Winname, 0, 255, nothing)
    cv.createTrackbar('S2', Winname, 0, 255, nothing)
    cv.createTrackbar('V2', Winname, 0, 255, nothing)

    while True:
        frame = cv.imread(example_img)
        frame = frame[0:35, 120:180]
        frame = cv_resize(frame, 35 * 10, 60 * 10)
        H = cv.getTrackbarPos('H', 'Frame:')
        S = cv.getTrackbarPos('S', 'Frame:')
        V = cv.getTrackbarPos('V', 'Frame:')
        H2 = cv.getTrackbarPos('H2', 'Frame:')
        S2 = cv.getTrackbarPos('S2', 'Frame:')
        V2 = cv.getTrackbarPos('V2', 'Frame:')
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        lower_boundary = np.array([H, S, V])
        upper_boundary = np.array([H2, S2, V2])
        mask = cv.inRange(hsv, lower_boundary, upper_boundary)
        final = cv.bitwise_and(frame, frame, mask=mask)
        cv.imshow("Frame:", final)

        if cv.waitKey(1) == ord('q'):
            break

    cv.destroyAllWindows()


if __name__ == "__main__":
    find_color_range()
