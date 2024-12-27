import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from model import PneumoniaCNN
from dataset import PneumoniaDataset

# 评估模型函数
def evaluate_model(model, test_loader, device='cuda'):
    model.to(device)
    model.eval()  # 切换到评估模式

    all_labels = []
    all_preds = []

    with torch.no_grad():  # 不需要计算梯度
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())  # 保存实际标签
            all_preds.extend(predicted.cpu().numpy())  # 保存预测标签

    # 计算各项评价指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    # 打印结果
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

# 数据加载：加载测试集
test_dataset = PneumoniaDataset(data_dir="D:/pythonProject1/Pneumonia/test")  # 确保测试集路径正确
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 加载训练好的模型
model = PneumoniaCNN()
model.load_state_dict(torch.load('D:/pythonProject1/Pneumonia/model.pth'))
model.eval()  # 切换到评估模式

# 使用测试集评估模型
evaluate_model(model, test_loader, device='cpu')  # 如果没有GPU可用，将device改为'cpu'

