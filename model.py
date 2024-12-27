import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义PneumoniaCNN模型
class PneumoniaCNN(nn.Module):
    def __init__(self):
        super(PneumoniaCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 输入1通道，输出32通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 输入32通道，输出64通道
        self.pool = nn.MaxPool2d(2, 2)  # 最大池化，2x2窗口
        self.fc1 = nn.Linear(64 * 56 * 56, 128)  # 假设输入图像为224x224
        self.fc2 = nn.Linear(128, 2)  # 2类：NORMAL, PNEUMONIA

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 卷积1 + ReLU + 池化
        x = self.pool(F.relu(self.conv2(x)))  # 卷积2 + ReLU + 池化
        x = torch.flatten(x, 1)  # 展平，准备输入全连接层
        x = F.relu(self.fc1(x))  # 全连接层1 + ReLU
        x = self.fc2(x)  # 输出层
        return x

# 创建模型实例
model = PneumoniaCNN()

# # 打印模型结构
# print("模型结构:")
# print(model)
#
# # 创建一个假数据输入来查看每一层的输出形状
# # 假设输入图像大小是224x224，通道数是1（灰度图像）
# sample_input = torch.randn(1, 1, 224, 224)  # batch_size=1, channel=1, height=224, width=224
#
# # 前向传播通过每一层，输出每层的形状
# x = sample_input
# print("\n输入图像形状:", x.shape)
#
# x = model.conv1(x)
# print("通过conv1后的形状:", x.shape)
#
# x = model.pool(F.relu(x))
# print("通过pool1后的形状:", x.shape)
#
# x = model.conv2(x)
# print("通过conv2后的形状:", x.shape)
#
# x = model.pool(F.relu(x))
# print("通过pool2后的形状:", x.shape)
#
# x = torch.flatten(x, 1)
# print("通过flatten后的形状:", x.shape)
#
# x = model.fc1(x)
# print("通过fc1后的形状:", x.shape)
#
# x = model.fc2(x)
# print("通过fc2后的形状:", x.shape)
