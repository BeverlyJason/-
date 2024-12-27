import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
from dataset import PneumoniaDataset  # 自定义数据集
from model import PneumoniaCNN  # 自定义模型
matplotlib.use('TkAgg')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 训练函数
def train_model(model, train_loader, val_loader, num_epochs=20, lr=0.001, batch_size=32, device='cuda'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 适用于分类任务
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 使用Adam优化器

    # 学习率调度器
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)  # 每5个epoch将学习率减少一半

    # 记录训练过程的准确率和损失
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # 训练过程
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # 训练阶段
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        # 计算训练集损失和准确率
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total * 100
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total * 100
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # 打印每个epoch的结果
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        # 更新学习率
        scheduler.step()

    return train_losses, val_losses, train_accuracies, val_accuracies


# 数据加载
train_dataset = PneumoniaDataset(data_dir="D:/pythonProject1/Pneumonia/train")
val_dataset = PneumoniaDataset(data_dir="D:/pythonProject1/Pneumonia/val")  # 需要提供验证集路径

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 创建模型实例
model = PneumoniaCNN()

# 开始训练
train_losses, val_losses, train_accuracies, val_accuracies = train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=20,
    lr=0.001,
    batch_size=32,
    device='cuda'  # 或者 'cpu'
)


# 绘制损失和准确率曲线
def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    # 绘制损失曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curve')

    plt.show()


# 绘制训练过程的图表
plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)


# 超参数调优
def hyperparameter_tuning():
    learning_rates = [0.001, 0.0001]
    batch_sizes = [16, 32]
    num_epochs = 10

    best_acc = 0
    best_params = {}

    for lr in learning_rates:
        for batch_size in batch_sizes:
            print(f"Training with lr={lr}, batch_size={batch_size}")
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # 重新创建模型
            model = PneumoniaCNN()

            # 训练模型
            _, _, _, val_accuracies = train_model(
                model,
                train_loader,
                val_loader,
                num_epochs=num_epochs,
                lr=lr,
                batch_size=batch_size,
                device='cuda'
            )

            # 记录最佳参数
            final_val_accuracy = val_accuracies[-1]
            if final_val_accuracy > best_acc:
                best_acc = final_val_accuracy
                best_params = {'lr': lr, 'batch_size': batch_size}

    print(f"Best parameters found: {best_params}, Validation Accuracy: {best_acc:.2f}%")

# 调用超参数调优
hyperparameter_tuning()
# 保存训练好的模型
torch.save(model.state_dict(), 'D:/pythonProject1/Pneumonia/model.pth')
