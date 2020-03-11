# -*- coding: utf-8 -*-
# @Time    : 20/02/13 11:51
# @Author  : Jay Lam
# @File    : pretreat.py
# @Software: PyCharm


import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# 以下为使用未经重写的DataLoader作为训练集加载器，可以复制到任何测试开头
data_transform = transforms.Compose([
    transforms.Resize(240),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
    # transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

train_dataset = datasets.ImageFolder(root="D:\\bubbleTracker_data\\test", transform=data_transform)
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=8,
                              shuffle=True,
                              num_workers=8,
                              drop_last=True)


# 将图片处理为灰度
def convert2Grey(path):
    for home, dirs, files in os.walk(path):
        for filename in files:
            fullname = os.path.join(home, filename)
            img = cv2.imread(fullname)
            img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(fullname, img2)
            print(fullname,"处理完成！")


# 以下为展示batch后的效果
def show_batch_image(sample_batch):
    labels_batch = sample_batch[1]
    images_batch = sample_batch[0]

    for i in range(8):
        label_ = labels_batch[i].item()
        image_ = np.transpose(images_batch[i], (1, 2, 0))
        ax = plt.subplot(1, 8, i + 1)
        ax.imshow(image_)
        ax.set_title(str(label_))
        ax.axis('off')
        plt.pause(0.01)


plt.figure()

if __name__ == '__main__':
    """
    for i_batch, sample_batch in enumerate(train_dataloader):
        show_batch_image(sample_batch)
        plt.show()
    """
    convert2Grey("E:\\Trans_01-20200121")
    print("全部处理完成！")
