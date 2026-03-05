# CNN 算法设计文档

## 1. 概述

卷积神经网络（Convolutional Neural Network, CNN）是一种专门用于处理具有网格结构数据的深度学习模型，特别适合处理图像数据。本项目实现了基础CNN和ResNet系列模型。

## 2. 模型结构

### 2.1 基础CNN模型

基础CNN模型由以下部分组成：
- **卷积层**：提取图像特征
- **池化层**：降低特征图维度
- **全连接层**：分类输出

具体结构如下：
```
BasicCNN(
  (features): Sequential(
    (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): ReLU(inplace=True)
    (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU(inplace=True)
    (8): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (classifier): Sequential(
    (0): Linear(in_features=2048, out_features=512, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=512, out_features=10, bias=True)
  )
)
```

### 2.2 ResNet模型

ResNet（Residual Network）引入了残差连接，解决了深层网络训练困难的问题。本项目实现了ResNet18和ResNet34模型。

ResNet的核心是残差块，结构如下：
```
ResNetBlock(
  (conv1): Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(stride, stride), padding=(1, 1), bias=False)
  (bn1): BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (conv2): Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (bn2): BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (shortcut): Sequential(
    (0): Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(stride, stride), bias=False)
    (1): BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
)
```

## 3. 实现细节

- **卷积操作**：使用`nn.Conv2d`实现，支持不同的 kernel size 和 stride
- **激活函数**：使用ReLU激活函数
- **池化操作**：使用最大池化（Max Pooling）
- **批量归一化**：使用`nn.BatchNorm2d`加速训练
- ** dropout**：在全连接层使用dropout防止过拟合

## 4. 训练策略

- **优化器**：使用Adam优化器
- **学习率**：初始学习率为0.001
- **批量大小**：默认批量大小为64
- **训练轮数**：默认训练10轮

## 5. 性能评估

使用CIFAR-10数据集进行评估，主要评估指标包括：
- 准确率（Accuracy）
- 精确率（Precision）
- 召回率（Recall）
- F1分数（F1 Score）

## 6. 使用示例

```python
from src.cnn.model import resnet18
import torch

# 初始化模型
model = resnet18(num_classes=10)

# 前向传播
input = torch.randn(1, 3, 32, 32)
output = model(input)
```