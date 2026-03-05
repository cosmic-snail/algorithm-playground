# Transformer 算法设计文档

## 1. 概述

Transformer是一种基于自注意力机制的深度学习模型，由Google在2017年提出，最初用于机器翻译任务。本项目实现了基于Transformer的分类模型。

## 2. 模型结构

### 2.1 核心组件

Transformer模型由以下核心组件组成：
- **位置编码（Positional Encoding）**：为输入序列添加位置信息
- **多头注意力（Multi-Head Attention）**：捕获不同子空间的注意力信息
- **前馈神经网络（Feed Forward Network）**：对注意力输出进行非线性变换
- **层归一化（Layer Normalization）**：加速模型收敛

### 2.2 编码器结构

Transformer编码器由多个相同的编码器层堆叠而成，每个编码器层包含：
1. 多头自注意力机制
2. 前馈神经网络
3. 残差连接和层归一化

### 2.3 分类器结构

本项目实现的Transformer分类器结构如下：
```
TransformerClassifier(
  (embedding): Embedding(vocab_size, d_model)
  (pos_encoder): PositionalEncoding()
  (encoder): TransformerEncoder(
    (layers): ModuleList(
      (0-5): 6 x TransformerEncoderLayer(
        (self_attn): MultiHeadAttention()
        (feed_forward): FeedForward()
        (norm1): LayerNorm()
        (norm2): LayerNorm()
        (dropout1): Dropout()
        (dropout2): Dropout()
      )
    )
  )
  (fc): Linear(d_model, num_classes)
  (dropout): Dropout()
)
```

## 3. 实现细节

### 3.1 位置编码

使用正弦和余弦函数生成位置编码：
```python
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
```

### 3.2 多头注意力

多头注意力将输入映射到多个子空间，然后在每个子空间中计算注意力，最后将结果拼接：
```python
# 线性变换
q = self.w_q(q).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
k = self.w_k(k).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
v = self.w_v(v).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)

# 计算注意力分数
attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
attn_weights = torch.softmax(attn_scores, dim=-1)

# 计算注意力输出
output = torch.matmul(attn_weights, v)
output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.nhead * self.d_k)
output = self.w_o(output)
```

### 3.3 前馈神经网络

前馈神经网络由两个线性层和一个ReLU激活函数组成：
```python
def forward(self, x):
    x = self.linear1(x)
    x = self.relu(x)
    x = self.linear2(x)
    return x
```

## 4. 训练策略

- **优化器**：使用Adam优化器
- **学习率**：初始学习率为0.001
- **批量大小**：默认批量大小为64
- **训练轮数**：默认训练10轮
- **dropout**：使用0.1的dropout率防止过拟合

## 5. 性能评估

使用CIFAR-10数据集进行评估，主要评估指标包括：
- 准确率（Accuracy）
- 精确率（Precision）
- 召回率（Recall）
- F1分数（F1 Score）

## 6. 使用示例

```python
from src.transformer.model import get_transformer_model
import torch

# 初始化模型
model = get_transformer_model(vocab_size=10000, num_classes=10)

# 前向传播
input = torch.randint(0, 10000, (1, 32))  # (batch_size, seq_len)
output, attn_list = model(input)
```