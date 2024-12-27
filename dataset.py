import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class PneumoniaDataset(Dataset):
    def __init__(self, data_dir):
        """
        初始化数据集，读取正常肺部图像和肺炎图像
        data_dir: 数据路径，包含NORMAL和PNEUMONIA子文件夹
        """
        self.data_dir = data_dir

        # 获取NORMAL和PNEUMONIA的图像路径
        self.normal_images = glob.glob(os.path.join(data_dir, 'NORMAL', '*.jpeg'))  # 假设图像是jpeg格式
        self.pneumonia_images = glob.glob(os.path.join(data_dir, 'PNEUMONIA', '*.jpeg'))

        # 将两类图像合并
        self.images = self.normal_images + self.pneumonia_images

    def __len__(self):
        # 返回数据集大小
        return len(self.images)

    def augment(self, image):
        """
        手动进行数据增强：这里使用了随机水平翻转和随机旋转
        """
        # 随机水平翻转
        if random.random() > 0.5:
            image = cv2.flip(image, 1)  # 1表示水平翻转

        # 随机旋转
        if random.random() > 0.5:
            angle = random.randint(-30, 30)  # 旋转角度范围
            rows, cols = image.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            image = cv2.warpAffine(image, M, (cols, rows))

        return image

    def __getitem__(self, idx):
        # 获取图像路径
        img_path = self.images[idx]
        # 读取图像
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图
        # 将图像调整为固定尺寸（比如224x224）
        img = cv2.resize(img, (224, 224))

        # 应用数据增强（翻转、旋转等）
        img = self.augment(img)

        # 创建标签，0代表NORMAL，1代表PNEUMONIA
        label = 0 if 'NORMAL' in img_path else 1

        # 归一化，将像素值缩放到[0, 1]范围
        img = img / 255.0

        # 将图像转换为PyTorch tensor，并调整维度为C H W (1,224,224)
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # 增加一个通道维度

        return img, label

# 数据加载
train_dataset = PneumoniaDataset(data_dir="D:/pythonProject1/Pneumonia/train")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

if __name__ == "__main__":
    isbi_dataset = PneumoniaDataset("D:/pythonProject1/Pneumonia/train")
    print("数据个数：", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=2,
                                               shuffle=False)  # C H W --> B C H W

    for img, label in train_loader:
        # 将图像从 PyTorch Tensor 转换为 NumPy 数组，以便使用 matplotlib 显示
        img_np = img.squeeze(1).numpy()  # 删除通道维度 (B, 1, H, W -> B, H, W)

        # 批次中的每个图像与标签
        for i in range(img.shape[0]):
            img_single = img_np[i]  # 取出当前的单个图像
            label_single = label[i].item()  # 获取当前图像的标签
            # 获取图像形状
            print(img.shape)
            print(label.shape)

            # 创建子图并显示图像和标签
            plt.figure(figsize=(6, 6))

            # 显示图像
            plt.imshow(img_single, cmap='gray')  # 显示灰度图像 (224, 224)
            plt.title(f"Label: {'NORMAL' if label_single == 0 else 'PNEUMONIA'}")
            plt.axis('off')  # 不显示坐标轴

            plt.show()