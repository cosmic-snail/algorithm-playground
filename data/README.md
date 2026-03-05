# 数据集说明

## 目录结构

```
data/
├── raw/            # 原始数据
├── processed/      # 处理后的数据
└── README.md       # 数据集说明
```

## 数据集格式

### CIFAR-10数据集

项目默认使用CIFAR-10数据集，训练脚本会自动下载。CIFAR-10包含10个类别，每个类别有6000张32x32的彩色图像。

### 自定义数据集

如果需要使用自定义数据集，请将数据放入`data/raw/`目录，并按照以下格式组织：

```
data/raw/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
└── test/
    ├── class1/
    ├── class2/
    └── ...
```

## 数据处理

1. 原始数据放入`data/raw/`目录
2. 运行数据处理脚本（如果需要）
3. 处理后的数据会保存在`data/processed/`目录

## 数据增强

在`scripts/train.py`中，可以通过修改`transform`变量来添加数据增强：

```python
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```