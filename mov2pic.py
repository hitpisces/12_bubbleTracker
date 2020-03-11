import os
import cv2

path1 = 'D:\\原10T\\气泡图像识别\\Trans_03-20200301\\'
path2 = 'E:\\'


def MkDir():
    dirs = os.listdir(path1)
    for dir in dirs:
        file_name = path2 + str(dir)
        # file_name2 = file_name.replace(".mp4", "");

        os.mkdir(file_name)
        each_video_save_full_path = os.path.join(path2, dir) + '/'
        print(each_video_save_full_path)
        each_video_full_path = os.path.join(path1, dir)
        print(each_video_full_path)
        cap = cv2.VideoCapture(each_video_full_path)

        frame_count = 1
        timeF = 30

        success = True

        while (success):
            success, frame = cap.read()
            # if success==True:
            if (frame_count % timeF == 0):
                # cv2.imwrite(each_video_save_full_path + str(frame_count) +".jpg" % frame_count,frame)
                cv2.imwrite(each_video_save_full_path + dir + "frame%d.jpg" % (frame_count / 30), frame)
            frame_count = frame_count + 1

        cap.release()


MkDir()
