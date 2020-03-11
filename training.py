# -*- coding: utf-8 -*-
# @Time    : 20/02/13 11:51
# @Author  : Jay Lam
# @File    : training.py
# @Software: PyCharm


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import datetime
import time


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


net = CNN_Net()

# 配置随机初始参数
random_state = 1
torch.manual_seed(random_state)
torch.rand(random_state).cuda()
np.random.seed(random_state)
epochs = 10  # 训练次数
use_gpu = torch.cuda.is_available()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("当前使用的GPU是：" + str(torch.cuda.get_device_name(torch.cuda.current_device())))
if use_gpu:
    net = CNN_Net().cuda()
else:
    net = CNN_Net()
print(net)

# 以下为使用未经重写的DataLoader作为训练集加载器，可以复制到任何测试
data_transform = transforms.Compose([
    transforms.Resize(240),
    transforms.CenterCrop(120),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
    # transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

train_dataset = datasets.ImageFolder(root="D:\\bubbleTracker_data\\train", transform=data_transform)
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=8,
                              shuffle=True,
                              num_workers=4,
                              drop_last=True)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)


def train():
    for epoch in range(epochs):
        running_loss = 0.0
        train_correct = 0.0
        train_total = 0
        for i, data in enumerate(train_dataloader, 0):
            inputs, train_labels = data
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(train_labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(train_labels)
            optimizer.zero_grad()
            outputs = net(inputs)
            _, train_predicted = torch.max(outputs.data, 1)
            # print(train_predicted)
            # print(labels)
            train_correct += (train_predicted == labels.data).sum()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_total += train_labels.size(0)
            print('train %d epoch loss: %.3f  acc: %.3f ' % (
                epoch + 1, running_loss / train_total, 100 * train_correct / train_total))


if __name__ == "__main__":
    start_time = time.time()
    train()
    torch.save(net.state_dict(),'CNN_Trained.pt')
    print("本次训练结束时间为：" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("本次训练耗时为：", round(time.time() - start_time, 2), "秒")
