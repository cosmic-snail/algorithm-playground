import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from src.cnn.model import resnet18
from src.transformer.model import get_transformer_model
from src.utils.utils import AverageMeter, accuracy, plot_training_curves, save_model, set_seed

parser = argparse.ArgumentParser(description='Model Training')
parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'transformer'], help='Model type')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--save-path', type=str, default='checkpoints/model.pth', help='Model save path')

args = parser.parse_args()

# 设置随机种子
set_seed(args.seed)

# 准备数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data/raw', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

valset = torchvision.datasets.CIFAR10(root='./data/raw', train=False, download=True, transform=transform)
valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=2)

# 初始化模型
if args.model == 'cnn':
    model = resnet18(num_classes=10)
elif args.model == 'transformer':
    # Transformer模型需要调整输入格式
    model = get_transformer_model(vocab_size=10000, num_classes=10)

# 移至GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# 训练循环
train_losses = []
train_accs = []
val_losses = []
val_accs = []

for epoch in range(args.epochs):
    # 训练阶段
    model.train()
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 前向传播
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = criterion(outputs, targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 计算准确率
        acc = accuracy(outputs, targets)[0]
        train_loss.update(loss.item(), inputs.size(0))
        train_acc.update(acc.item(), inputs.size(0))
    
    # 验证阶段
    model.eval()
    val_loss = AverageMeter()
    val_acc = AverageMeter()
    
    with torch.no_grad():
        for inputs, targets in valloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, targets)
            acc = accuracy(outputs, targets)[0]
            val_loss.update(loss.item(), inputs.size(0))
            val_acc.update(acc.item(), inputs.size(0))
    
    # 记录结果
    train_losses.append(train_loss.avg)
    train_accs.append(train_acc.avg)
    val_losses.append(val_loss.avg)
    val_accs.append(val_acc.avg)
    
    print(f'Epoch [{epoch+1}/{args.epochs}], '  
          f'Train Loss: {train_loss.avg:.4f}, Train Acc: {train_acc.avg:.2f}%, '  
          f'Val Loss: {val_loss.avg:.4f}, Val Acc: {val_acc.avg:.2f}%')

# 保存模型
import os
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
save_model(model, args.save_path)

# 绘制训练曲线
plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path='plots/training_curves.png')

print('Training completed!')