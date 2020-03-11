import cv2
import os.path

# 使用opencv按一定间隔截取视频帧，并保存为图片

filepath = "D:\\原10T\\气泡图像识别\\Trans_03-20200301"  # 视频所在文件夹,将需要切分的所有视频都放在这个文件夹
pathDir = os.listdir(filepath)
a = 1  # 图片计数
for allDir in pathDir:
    videopath = r"C:/pic/" + allDir

    vc = cv2.VideoCapture(videopath)  # 读入视频文件

    c = 1

    if vc.isOpened():  # 判断是否正常打开
        rval, frame = vc.read()

    else:
        rval = False

    timeF = 30  # 视频帧计数间隔频率，根据视频决定

    while rval:  # 循环读取视频帧
        rval, frame = vc.read()
        if c % timeF == 0:  # 每隔timeF帧进行存储操作
            cv2.imwrite("C:/pic/" + str(a) + '.jpg', frame)  # 视频存储路径，可以自定义

            a = a + 1

        c = c + 1
        cv2.waitKey(1)

    vc.release()
