# -*- coding: utf-8 -*-
# @Time    : 20/2/10 16:52
# @Author  : Jay Lam
# @File    : bubbleTrackerStatic.py
# @Software: PyCharm


import sys
import cv2
import numpy as np
from math import *

pic = cv2.imread("b1.png")
pic2 = cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)
ret,binary = cv2.threshold(pic2, 135, 255, cv2.THRESH_BINARY)
circles = cv2.HoughCircles(binary, cv2.HOUGH_GRADIENT, 1, 25, param1 = 30, param2 = 10, minRadius = 5, maxRadius = 20)

if circles is not None:
    for i in circles[0]:
        cv2.circle(pic2, (i[0], i[1]), i[2], (255, 0, 0), 1)

cv2.imshow("test",pic2)
cv2.waitKey(0)