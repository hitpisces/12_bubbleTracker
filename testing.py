# -*- coding: utf-8 -*-
# @Time    :  21:53
# @Author  : Jay Lam
# @File    : testing.py
# @Software: PyCharm


import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time
import datetime


target_label_names = ['0', '1', '2', '3', '4', '5', '6', '7']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 构建测试数据集
test_transform = transforms.Compose([
    transforms.Resize(240),
    transforms.CenterCrop(120),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
    # transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

test_dataset = datasets.ImageFolder(root="E:\\Trans_01-20200121",
                                    transform=test_transform)
test_dataloader = DataLoader(dataset=test_dataset,
                             batch_size=8,
                             shuffle=False,
                             num_workers=4,
                             drop_last=True)


# 网络模型
class CNN_Net(nn.Module):
    """
    CNN计算

    (H - k +2 * P) / S + 1
    (W - k +2 * P) / S + 1

    LetNet-5
    input: 32*32*3

    out_conv1 = (32-5)+1 = 28
    max_pool1 = 28 / 2 = 14
    out_conv2 = (14 - 5) + 1 = 10
    max_pool2 = 10 / 2 = 5
    """

    def __init__(self):
        super(CNN_Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 8, 5)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 8, 5)
        self.fc1 = nn.Linear(8 * 27 * 27, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 8)

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = x.view(-1, 8 * 27 * 27)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


CNN_Net().load_state_dict(torch.load('CNN_Trained.pt', map_location=device))  # CNN_Net后面要加（）
net = CNN_Net()


# 加载模型到GPU并在整个测试集上测试，评价指标为acc
def test_net():
    correct = 0
    total = 0
    count = 0
    print("当前使用的GPU是：" + str(torch.cuda.get_device_name(torch.cuda.current_device())))

    with torch.no_grad():
        for sample_batch in test_dataloader:
            images = sample_batch[0]
            labels = sample_batch[1]

            out = net(images)
            _, prediction = torch.max(out, 1)
            correct += (prediction == labels).sum().item()
            total += labels.size(0)
            print('batch:{}'.format(count + 1), 'correct number in each batch :{}'.format(correct))
            count += 1

    accuracy = float(correct) / total
    print('Acc ={:.5f}'.format(accuracy))


if __name__ == "__main__":
    start_time = time.time()
    test_net()
    print("本次测试结束时间为：" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("本次测试耗时为：", time.time() - start_time, "秒")
