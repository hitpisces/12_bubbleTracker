# -*- coding: utf-8 -*-
# @Time    : 19/11/12 11:51
# @Author  : Jay Lam
# @File    : moveFiles2Dir.py
# @Software: PyCharm


import os
import shutil
import random

originalDir = "E:\\气泡图像识别\\Trans_01-20200121\\pic-0.26-0.16"
targetDir = "D:\\bubbleTracker_data\\train\\0.3"

fileCounter = len(os.listdir(originalDir))
fileList = os.listdir(originalDir)
i = 0
for filename in fileList:
    os.rename(originalDir + "\\" + filename, originalDir + "\\" + str(i) + ".png")
    i = i + 1
    print("正在修改文件"+filename)
print("修改名称完成！")

ratio = 0.5  # 自定义抽取比例
pickNumber = int(fileCounter * ratio)
fileList2 = os.listdir(originalDir)
samplePic = random.sample(fileList2, pickNumber)
print(fileCounter)
print(samplePic)
for files in samplePic:
    shutil.move(originalDir+"\\"+files, targetDir)
    print("正在移动：" + files)
print("完成随机移动文件！")
