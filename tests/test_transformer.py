import torch
from src.transformer.model import get_transformer_model, PositionalEncoding, MultiHeadAttention

def test_positional_encoding():
    """测试位置编码"""
    d_model = 512
    max_seq_len = 100
    pe = PositionalEncoding(d_model, max_seq_len)
    input = torch.randn(10, 32, d_model)  # (seq_len, batch_size, d_model)
    output = pe(input)
    assert output.shape == input.shape, f"Expected output shape {input.shape}, got {output.shape}"
    print("PositionalEncoding test passed!")

def test_multi_head_attention():
    """测试多头注意力"""
    d_model = 512
    nhead = 8
    attn = MultiHeadAttention(d_model, nhead)
    q = torch.randn(1, 10, d_model)  # (batch_size, seq_len, d_model)
    k = torch.randn(1, 10, d_model)
    v = torch.randn(1, 10, d_model)
    output, attn_weights = attn(q, k, v)
    assert output.shape == q.shape, f"Expected output shape {q.shape}, got {output.shape}"
    assert attn_weights.shape == (1, nhead, 10, 10), f"Expected attn_weights shape (1, {nhead}, 10, 10), got {attn_weights.shape}"
    print("MultiHeadAttention test passed!")

def test_transformer_model():
    """测试Transformer模型"""
    model = get_transformer_model(vocab_size=10000, num_classes=10)
    input = torch.randint(0, 10000, (1, 32))  # (batch_size, seq_len)
    output, attn_list = model(input)
    assert output.shape == (1, 10), f"Expected output shape (1, 10), got {output.shape}"
    assert len(attn_list) == 6, f"Expected 6 attention layers, got {len(attn_list)}"
    print("Transformer model test passed!")

if __name__ == "__main__":
    test_positional_encoding()
    test_multi_head_attention()
    test_transformer_model()
    print("All Transformer tests passed!")