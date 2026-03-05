import torch
from src.cnn.model import BasicCNN, resnet18

def test_basic_cnn():
    """测试基础CNN模型"""
    model = BasicCNN(num_classes=10)
    input = torch.randn(1, 3, 32, 32)
    output = model(input)
    assert output.shape == (1, 10), f"Expected output shape (1, 10), got {output.shape}"
    print("BasicCNN test passed!")

def test_resnet18():
    """测试ResNet18模型"""
    model = resnet18(num_classes=10)
    input = torch.randn(1, 3, 32, 32)
    output = model(input)
    assert output.shape == (1, 10), f"Expected output shape (1, 10), got {output.shape}"
    print("ResNet18 test passed!")

if __name__ == "__main__":
    test_basic_cnn()
    test_resnet18()
    print("All CNN tests passed!")