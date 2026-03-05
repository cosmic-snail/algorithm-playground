# algorithm-playground

## 项目介绍

本项目用于验证CNN、Transformer等深度学习算法，包含完整的目录结构和示例代码。

## 目录结构

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

## 快速开始

1. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

2. 准备数据集
   - 将数据集放入 `data/raw/` 目录
   - 运行数据处理脚本

3. 训练模型
   ```bash
   python scripts/train.py
   ```

4. 评估模型
   ```bash
   python scripts/evaluate.py
   ```

5. 运行测试
   ```bash
   python -m pytest tests/
   ```
My personal algorithm playground: learn by coding, validate by testing, grow step by step
