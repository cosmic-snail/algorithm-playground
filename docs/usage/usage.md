# 使用文档

## 1. 环境搭建

### 1.1 安装依赖

```bash
pip install -r requirements.txt
```

### 1.2 准备数据集

项目默认使用CIFAR-10数据集，训练脚本会自动下载。如果需要使用其他数据集，请将数据放入`data/raw/`目录，并修改相应的数据集加载代码。

## 2. 训练模型

### 2.1 训练CNN模型

```bash
python scripts/train.py --model cnn --epochs 10 --batch-size 64 --lr 0.001
```

### 2.2 训练Transformer模型

```bash
python scripts/train.py --model transformer --epochs 10 --batch-size 64 --lr 0.001
```

### 2.3 训练参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --model | 模型类型 (cnn/transformer) | cnn |
| --epochs | 训练轮数 | 10 |
| --batch-size | 批量大小 | 64 |
| --lr | 学习率 | 0.001 |
| --seed | 随机种子 | 42 |
| --save-path | 模型保存路径 | checkpoints/model.pth |

## 3. 评估模型

### 3.1 评估CNN模型

```bash
python scripts/evaluate.py --model cnn --model-path checkpoints/model.pth
```

### 3.2 评估Transformer模型

```bash
python scripts/evaluate.py --model transformer --model-path checkpoints/model.pth
```

### 3.3 评估参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --model | 模型类型 (cnn/transformer) | cnn |
| --model-path | 模型路径 | checkpoints/model.pth |
| --batch-size | 批量大小 | 64 |

## 4. 运行测试

### 4.1 测试CNN模型

```bash
python tests/test_cnn.py
```

### 4.2 测试Transformer模型

```bash
python tests/test_transformer.py
```

### 4.3 使用pytest运行所有测试

```bash
python -m pytest tests/
```

## 5. 项目结构说明

```
algorithm-playground/
├── src/                 # 核心算法代码
│   ├── cnn/            # CNN相关算法
│   ├── transformer/    # Transformer相关算法
│   └── utils/          # 工具函数
├── data/               # 数据集
│   ├── raw/            # 原始数据
│   ├── processed/      # 处理后的数据
│   └── README.md       # 数据集说明
├── docs/               # 文档
│   ├── algorithm_design/  # 算法设计文档
│   ├── usage/          # 使用文档
│   └── README.md       # 文档说明
├── tests/              # 测试程序
│   ├── test_cnn.py     # CNN测试
│   ├── test_transformer.py  # Transformer测试
│   └── README.md       # 测试说明
├── scripts/            # 脚本
│   ├── train.py        # 训练脚本
│   ├── evaluate.py     # 评估脚本
│   └── README.md       # 脚本说明
├── requirements.txt    # 依赖包
└── README.md           # 项目说明
```

## 6. 自定义模型

### 6.1 自定义CNN模型

在`src/cnn/model.py`中添加新的CNN模型，例如：

```python
class CustomCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomCNN, self).__init__()
        # 自定义模型结构
    
    def forward(self, x):
        # 前向传播逻辑
        return x
```

### 6.2 自定义Transformer模型

在`src/transformer/model.py`中添加新的Transformer模型，例如：

```python
class CustomTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_classes):
        super(CustomTransformer, self).__init__()
        # 自定义模型结构
    
    def forward(self, x):
        # 前向传播逻辑
        return x
```

## 7. 常见问题

### 7.1 显存不足

如果遇到显存不足的问题，可以尝试：
- 减小批量大小
- 使用更小的模型
- 使用混合精度训练

### 7.2 训练过拟合

如果遇到过拟合问题，可以尝试：
- 增加dropout率
- 数据增强
- 早停策略
- 正则化

### 7.3 模型性能不佳

如果模型性能不佳，可以尝试：
- 调整学习率
- 使用不同的优化器
- 增加模型深度或宽度
- 调整超参数