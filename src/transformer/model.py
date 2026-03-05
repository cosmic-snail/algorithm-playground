import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_seq_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model, nhead):
        super(MultiHeadAttention, self).__init__()
        assert d_model % nhead == 0
        self.d_k = d_model // nhead
        self.nhead = nhead
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 线性变换
        q = self.w_q(q).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # 计算注意力输出
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.nhead * self.d_k)
        output = self.w_o(output)
        
        return output, attn_weights

class FeedForward(nn.Module):
    """前馈神经网络"""
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    def __init__(self, d_model, nhead, d_ff=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src, src_mask=None):
        # 自注意力
        src2, attn = self.self_attn(src, src, src, src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # 前馈网络
        src2 = self.feed_forward(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src, attn

class TransformerEncoder(nn.Module):
    """Transformer编码器"""
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
    
    def forward(self, src, src_mask=None):
        output = src
        attn_list = []
        for layer in self.layers:
            output, attn = layer(output, src_mask)
            attn_list.append(attn)
        return output, attn_list

class TransformerClassifier(nn.Module):
    """基于Transformer的分类器"""
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward, num_classes, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, src_mask=None):
        src = self.embedding(src)
        src = self.dropout(src)
        src = self.pos_encoder(src)
        output, attn_list = self.encoder(src, src_mask)
        output = output.mean(dim=1)  # 平均池化
        output = self.fc(output)
        return output, attn_list

def get_transformer_model(vocab_size=10000, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, num_classes=10):
    """获取Transformer模型"""
    return TransformerClassifier(vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward, num_classes)