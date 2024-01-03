#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
# ==========================================
# Copyright 2024 Yang 
# PythonProjects - main
# ==========================================
#
#
# 
"""
import re
import shutil
import json
from imutils import contours as cnt

from global_settings import os, datetime, CAMERA_DIR, TXT_PATH, SAVE_DIR, TEMPLATE_RESULTS_PATH
from basic_funcs import cv, np, bounding_rect, cv_show, cv_resize, write


def get_matched_results(src_img_path, resize=True, show_img=False):
    """
    :param src_img_path:
    :param resize:
    :param show_img:
    :return:
    """
    if os.path.isfile(src_img_path):
        src_img = cv.imread(src_img_path)
    else:
        src_img = src_img_path
    if resize:
        # 更改图尺寸
        src_img = src_img[5:35, 120:180]
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
    # 轮廓
    contours, hierarchy = cv.findContours(binary_img.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
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
            inRange_img = cv.rectangle(inRange_img, (x, y), (x+w, y+h), (255, 0, 0), 3)

    locs = sorted(locs, key=lambda x: x[0])

    output = []
    # 遍历每一个轮廓中的数字
    with open(TEMPLATE_RESULTS_PATH, "r") as f:
        digits = json.loads(f.read())
    for (i, (gX, gY, gW, gH)) in enumerate(locs):
        # initialize the list of group digits
        groupOutput = []
        # 根据坐标提取每一个组
        group = binary_img[gY:gY + gH, gX:gX + gW]
        # cv_show('group',group)
        # 预处理
        group = cv.threshold(group, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
        # cv_show('group', group)
        # 计算每一组的轮廓
        digitCnts, hierarchy = cv.findContours(group.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # 计算每一组中的每一个数值
        for c in digitCnts:
            # 找到当前数值的轮廓，resize成合适的的大小
            (x, y, w, h) = bounding_rect(c)
            roi = group[y:y + h, x:x + w]
            roi = cv.resize(roi, (100, 100))
            # cv_show('roi', roi)

            # 计算匹配得分
            scores = []
            index = []

            # 在模板中计算每一个得分
            for (digit, digitROI) in digits.items():
                digit = re.findall(r"\d+", digit)[0]
                # 模板匹配
                result = cv.matchTemplate(roi, np.array(digitROI, dtype=roi.dtype), cv.TM_CCOEFF)
                (_, score, _, _) = cv.minMaxLoc(result)
                scores.append(score)
                index.append(digit)
            # 得到最合适的数字
            groupOutput.append(str(index[np.argmax(scores)]))
        output.append("".join(groupOutput))

    while show_img:
        cv_show('show_img', draw_img)
        cv_show('show_yellow', inRange_img)
        cv_show('show_gray', gray_img)
        cv_show('show_binary', binary_img)

        if cv.waitKey(1) == ord('q'):
            break
    cv.destroyAllWindows()

    # return gray_img
    return "".join(output), inRange_img


def get_binary_img(img, resize=True):
    """
    :param img:
    :param resize:
    :return:
    """
    if os.path.isfile(img):
        src_img = cv.imread(img)
    else:
        src_img = img
    if resize:
        # 更改图尺寸
        src_img = src_img[5:35, 120:180]
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

    return binary_img


def get_rect(img):
    """

    :param img: UMat, binary_img
    :return:
    """
    binary_img = img.copy()
    # 轮廓
    contours, hierarchy = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        contours = cnt.sort_contours(contours, method="left-to-right")[0]
    locs = []
    for (i, c) in enumerate(contours):
        # 获取矩形
        (x, y, w, h) = bounding_rect(c)
        # 添加矩形
        if h > 300:
            locs.append((x, y, w, h))
            img = cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)

    locs = sorted(locs, key=lambda x: x[0])

    cnts = []
    for (i, (gX, gY, gW, gH)) in enumerate(locs):
        # 根据坐标提取每一个组
        group = binary_img[gY:gY + gH, gX:gX + gW]
        # cv_show('group',group)
        # 预处理
        group = cv.threshold(group, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
        # cv_show('group', group)
        # 计算每一组的轮廓
        digitCnts, hierarchy = cv.findContours(group.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts.append({"img": group, "cnts": digitCnts})

    return cnts, img


def recognize(cnts: list, template: dict):
    groupOutput = []
    for each in cnts:
        c = each['cnts']
        group = each['img']
        # 找到当前数值的轮廓，resize成合适的的大小
        (x, y, w, h) = bounding_rect(c)
        roi = group[y:y + h, x:x + w]
        roi = cv.resize(roi, (100, 100))
        # cv_show('roi', roi)

        # 计算匹配得分
        scores = []

        # 在模板中计算每一个得分
        for (digit, digitROI) in template.items():
            # 模板匹配
            result = cv.matchTemplate(roi, np.array(digitROI, dtype=roi.dtype), cv.TM_CCOEFF)
            (_, score, _, _) = cv.minMaxLoc(result)
            scores.append(score)
        # 得到最合适的数字
        groupOutput.append(str(np.argmax(scores)))

    return groupOutput


def capture(camera_index=1, do_ocr=True):
    print('开始运行')
    cap = cv.VideoCapture(camera_index, cv.CAP_DSHOW)  # 调用电脑摄像头
    cap.set(cv.CAP_PROP_FPS, 30)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    n = 1
    while True:

        ret, frame = cap.read()

        if ret:
            # frame = frame[0:100, 1400:1700]
            temp, inrange_frame = get_matched_results(frame, resize=True, show_img=False)
            cv.imshow('camera', inrange_frame)  # 生成摄像头窗口
            # 获取当前时间
            now_time = datetime.datetime.now()
            filename = datetime.datetime.strftime(now_time, "WIN_%Y%m%d_%H_%M_%S_Pro.jpg")
            cv.imwrite(os.path.join(CAMERA_DIR, filename), frame)
            print(f"{n = }, {filename = }")

            if do_ocr:
                filenames = next(os.walk(CAMERA_DIR), (None, None, []))[2]  # [] if no file
                for each in filenames:
                    dt = re.findall(r"\d{2}", each)
                    dt[4] = str(int(dt[4]) - 1)
                    dt[4] = "0" + dt[4] if len(dt[4]) == 1 else dt[4]
                    save_path = os.path.join(SAVE_DIR, f"{dt[0]}{dt[1]}-{dt[2]}-{dt[3]}")
                    if each.startswith("WIN_202") and each.endswith(".jpg") and not os.path.isfile(
                            os.path.join(save_path, each)):
                        # move file
                        if not os.path.exists(save_path):
                            os.mkdir(save_path)
                        shutil.move(
                            os.path.join(CAMERA_DIR, each),
                            os.path.join(save_path, each)
                        )
                        text = "{0}{1}-{2}-{3}T{4}:{5}:{6}Z;{7};".format(*dt, temp)
                        write(TXT_PATH, f"{text}\n")
                        print(text)

        n += 1

        if cv.waitKey(5000) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


def _ocr_dir(img_dir, results_path="", show_img=False, save_results=True):
    filenames = next(os.walk(img_dir), (None, None, []))[2]  # [] if no file
    for each in filenames:

        dt = re.findall(r"\d{2}", each)
        dt[4] = str(int(dt[4]) - 1 if int(dt[4]) > 0 else dt[4])
        dt[4] = "0" + dt[4] if len(dt[4]) == 1 else dt[4]

        temp, inrange_frame = get_matched_results(os.path.join(img_dir, each), resize=True, show_img=show_img)
        text = "{0}{1}-{2}-{3}T{4}:{5}:{6}Z;{7};".format(*dt, temp)
        print(text)
        if save_results and os.path.exists(results_path):
            write(results_path, f"{text}\n")


if __name__ == "__main__":
    # capture(camera_index=1)
    # img_dir = r"D:\PythonProjects\TempMeterRecognition\statics\Test"
    # results_path = r"D:\PythonProjects\TempMeterRecognition\inside_temperature - Copy.txt"
    img_dir = r"D:\Saved Pictures\2024-01-02"
    results_path = r"D:\Saved Pictures\inside_temperature.txt"

    _ocr_dir(img_dir, results_path, show_img=False, save_results=True)

