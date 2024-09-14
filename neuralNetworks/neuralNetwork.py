
import torch

# 神经网络的典型训练过程如下：

# 定义包含一些可学习的参数(或者叫权重)神经网络模型；
# 在数据集上迭代；
# 通过神经网络处理输入；
# 计算损失(输出结果和正确值的差值大小)；
# 将梯度反向传播回网络的参数；
# 更新网络的参数，主要使用如下简单的更新原则： weight = weight - learning_rate * gradient

import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义了一个名为 Net 的类，它继承自 nn.Module，这是所有神经网络模块的基类。
class Net(nn.Module):
    # 是类的构造函数，用于初始化网络层
    def __init__(self):
        # 调用父类的构造函数
        super(Net, self).__init__()
        # 创建一个卷积核，它有1个输入通道，6个输出通道，卷积核大小为 5*5
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 创建一个卷积核，它有6个输入通道，16个输出通道，卷积核大小为 5*5
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b 创建第一个全连接层，它将卷积层的输出映射到120个特征。
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 创建第二个全连接层，它将120个特征映射到84个特征。
        self.fc2 = nn.Linear(120, 84)
        # 创建第三个全连接层，它将84个特征映射到10个输出，通常对应于10类分类任务。
        self.fc3 = nn.Linear(84, 10)

    # 定义了数据通过网络的前向传播路径
    def forward(self, x):
        # 通过第一个卷积层，然后应用 ReLU 激活函数，接着进行最大池化，池化窗口大小为2x2。
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 通过第二个卷积层，然后应用 ReLU 激活函数，接着进行最大池化，池化窗口大小为2。
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # 将卷积层的输出展平，以匹配全连接层的输入要求。
        x = x.view(-1, self.num_flat_features(x))
        # 通过第一个全连接层，然后应用ReLU激活函数
        x = F.relu(self.fc1(x))
        # 通过第二个全连接层，然后应用ReLU激活函数
        x = F.relu(self.fc2(x))
        # 通过第三个全连接层，得到最终的输出
        x = self.fc3(x)
        return x

    # 是一个辅助函数，用于计算展平后的张量的特征数
    def num_flat_features(self, x):
        size = x.size()[1:]  # 获取除了批量维度以外的所有维度
        num_features = 1    # 初始化特征数
        for s in size:      # 遍历每个维度的大小
            num_features *= s   # 计算总的特征数
        return num_features

# 实例化 Net 类，创建网络模型
net = Net()
print(net)