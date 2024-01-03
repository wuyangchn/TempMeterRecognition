#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
# ==========================================
# Copyright 2024 Yang 
# PythonProjects - get_template
# ==========================================
#
#
# 
"""
import json
import os
import re

from imutils import contours as cnt

from basic_funcs import cv_resize, bounding_rect, cv_show, cv, np, write


def get_template(template_img_dir="", resize=True, show_img=False):

    if template_img_dir == "":
        raise NotADirectoryError

    digits = {}

    for each_img_lable in next(os.walk(template_img_dir), (None, None, []))[2]:  # [] if no file

        if not each_img_lable.endswith(".jpg"):
            continue

        number = re.findall(r"\d+", each_img_lable)[0]

        src_img = cv.imread(os.path.join(template_img_dir, each_img_lable))

        if resize:
            # 更改图尺寸
            # src_img = src_img[5:35, 120:180]
            src_img = cv_resize(src_img, 30 * 10, 60 * 10)

        hsv_img = cv.cvtColor(src_img, cv.COLOR_BGR2HSV)
        colors = [np.array([0, 0, 200]), np.array([255, 255, 255])]
        mask = cv.inRange(hsv_img, *colors)
        # 特殊颜色部分
        inRange_img = cv.bitwise_and(src_img, src_img, mask=mask)
        # 灰度图
        gray_img = cv.cvtColor(inRange_img, cv.COLOR_BGR2GRAY)
        # 二值图
        _, binary_img = cv.threshold(gray_img, 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY)
        # 计算轮廓
        contours, hierarchy = cv.findContours(binary_img.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        contours = cnt.sort_contours(contours, method="left-to-right")[0]
        # 绘制数字轮廓
        # -1 signifies drawing all contours
        draw_img = cv.drawContours(src_img, contours, -1, (0, 0, 255), 1)
        # 遍历轮廓
        locs = []
        for (i, c) in enumerate(contours):
            # 获取矩形
            (x, y, w, h) = bounding_rect(c)
            # 添加矩形
            if h > 300:
                locs.append((x, y, w, h))
                inRange_img = cv.rectangle(inRange_img, (x, y), (x + w, y + h), (255, 0, 0), 3)

        locs = sorted(locs, key=lambda x: x[0])

        (gX, gY, gW, gH) = locs[0]
        # 根据坐标提取每一个组
        group = binary_img[gY:gY + gH, gX:gX + gW]
        # cv_show('group',group)
        # 预处理
        group = cv.threshold(group, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
        # 计算每一组的轮廓
        digitCnts, hierarchy = cv.findContours(group.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # 找到当前数值的轮廓，resize成合适的的大小
        (x, y, w, h) = bounding_rect(digitCnts[0])
        roi = group[y:y + h, x:x + w]
        roi = cv.resize(roi, (100, 100))
        digits[each_img_lable] = roi.tolist()

        while show_img:
            cv_show('show_img', draw_img)
            cv_show('show_yellow', inRange_img)
            cv_show('show_gray', gray_img)
            cv_show('show_binary', binary_img)
            cv_show('group', group)
            cv_show('roi', roi)

            if cv.waitKey(1) == ord('q'):
                break
        cv.destroyAllWindows()

    return digits


if __name__ == "__main__":
    from global_settings import TEMPLATE2_DIR, TEMPLATE2_RESULTS_PATH
    digits_temp = get_template(template_img_dir=TEMPLATE2_DIR, resize=True, show_img=True)
    with open(TEMPLATE2_RESULTS_PATH, "w") as f:
        f.writelines(json.dumps(digits_temp))
