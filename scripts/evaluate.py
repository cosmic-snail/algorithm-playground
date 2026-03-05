import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from src.cnn.model import resnet18
from src.transformer.model import get_transformer_model
from src.utils.utils import evaluate_metrics, load_model

parser = argparse.ArgumentParser(description='Model Evaluation')
parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'transformer'], help='Model type')
parser.add_argument('--model-path', type=str, default='checkpoints/model.pth', help='Model path')
parser.add_argument('--batch-size', type=int, default=64, help='Batch size')

args = parser.parse_args()

# 准备数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

testset = torchvision.datasets.CIFAR10(root='./data/raw', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

# 初始化模型
if args.model == 'cnn':
    model = resnet18(num_classes=10)
elif args.model == 'transformer':
    model = get_transformer_model(vocab_size=10000, num_classes=10)

# 加载模型
model = load_model(model, args.model_path)

# 移至GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 评估模型
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for inputs, targets in testloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        _, predicted = torch.max(outputs, 1)
        y_true.extend(targets.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# 计算评估指标
metrics = evaluate_metrics(y_true, y_pred)

print('Evaluation Results:')
for metric, value in metrics.items():
    print(f'{metric}: {value:.4f}')

print('Evaluation completed!')