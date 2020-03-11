# -*- coding: utf-8 -*-
# @Time    : 20/2/2 15:28
# @Author  : Jay Lam
# @File    : testOpenCV.py
# @Software: PyCharm


import cv2
import time

if __name__ == '__main__':
    s = time.time()
    img = cv2.imread("20-01-19-18-10-01.png")
    print("imread:", time.time() - s)

    s2 = time.time()
    img2 = cv2.resize(img, (300, 400))
    print("resize:", time.time() - s2)

    s3 = time.time()
    img3 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("cvtColor:", time.time() - s3)

    #print(cv2.getBuildInformation())
    #print(cv2.__version__)
    cv2.imwrite("test4.png", img3)
