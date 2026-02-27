#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.models import resnet as ResNet
from torchvision.models import inception_v3
from torch.utils.data import DataLoader, Dataset
import yaml
import numpy as np

# Auto transfer to NPU
import torch_npu
from torch_npu.contrib import transfer_to_npu


with open('./config.yaml', 'r', encoding='utf-8') as f_config:
    config_result = f_config.read()
    config = yaml.load(config_result, Loader=yaml.FullLoader)

# 定义是否使用NPU
if config["train"]["is_gpu"]:
    if torch.npu.is_available():
        device = torch.device("npu")
        print("Using NPU for training")
    else:
        raise Exception("NPU不可用，请检查CANN环境")
else:
    device = torch.device("cpu")
    print("Using CPU for training")

# 检查模型保存地址
if not os.path.exists(config["train"]["out_model_path"]):
    os.makedirs(config["train"]["out_model_path"], exist_ok=True)
    print(f"Created directory: {config['train']['out_model_path']}")


# 准备数据集并预处理
transform_train = transforms.Compose([
    transforms.Resize((config["width"], config["height"])),
    transforms.RandomHorizontalFlip(0.5 if config["train"]["rotating"] else 0),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((config["width"], config["height"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

training_dir = config["train"]["train_data"]
train_dataset = torchvision.datasets.CIFAR10(
    root=training_dir,
    train=True,
    download=False,
    transform=transform_train
)

test_dir = config["train"]["test_data"]
test_dataset = torchvision.datasets.CIFAR10(
    root=test_dir,
    train=False,
    download=False,
    transform=transform_test
)

# 检查数据样本
print("=" * 50)
print("数据检查:")
sample_img, sample_label = train_dataset[0]
print(f"样本图像类型: {type(sample_img)}")
print(f"样本图像形状: {sample_img.shape}")
print(f"样本图像dtype: {sample_img.dtype}")
print(f"样本图像取值范围: [{sample_img.min():.3f}, {sample_img.max():.3f}]")
print(f"样本标签: {sample_label}")
print(f"类别数: {len(train_dataset.classes)}")
print(f"Batch size: {config['train']['batch_size']}")
print("=" * 50)

# DataLoader
trainloader = DataLoader(dataset=train_dataset, batch_size=config["train"]["batch_size"], 
                         shuffle=True, num_workers=0)  # num_workers=0 避免多进程问题
testloader = DataLoader(dataset=test_dataset, batch_size=config["train"]["batch_size"], 
                        shuffle=True, num_workers=0)

# 检查一个batch的数据
print("检查第一个batch:")
for batch_idx, (inputs, labels) in enumerate(trainloader):
    print(f"Batch {batch_idx}:")
    print(f"  inputs shape: {inputs.shape}")
    print(f"  inputs dtype: {inputs.dtype}")
    print(f"  inputs device: {inputs.device}")
    print(f"  labels shape: {labels.shape}")
    print(f"  labels dtype: {labels.dtype}")
    print(f"  inputs 取值范围: [{inputs.min():.3f}, {inputs.max():.3f}]")
    break  # 只检查第一个batch
print("=" * 50)

# 模型定义 - 先用ResNet18测试，InceptionV3可能有问题
classes = train_dataset.classes
num_classes = len(classes)

print(f"创建模型: {config['net']}")
if config["net"] == "ResNet18":
    net = ResNet.resnet18(num_classes=num_classes)
elif config["net"] == "ResNet34":
    net = ResNet.resnet34(num_classes=num_classes)
elif config["net"] == "ResNet50":
    net = ResNet.resnet50(num_classes=num_classes)
elif config["net"] == "InceptionV3":
    # 注意：InceptionV3需要aux_logits=False且对输入尺寸敏感
    net = inception_v3(num_classes=num_classes, aux_logits=False, init_weights=True)
else:
    raise Exception(f"不支持的模型: {config['net']}")

# 移动到设备
net = net.to(device)
print(f"Model loaded on {device}")

# 测试前向传播
print("测试前向传播...")
with torch.no_grad():
    test_input = torch.randn(2, 3, config["height"], config["width"]).to(device)
    print(f"测试输入形状: {test_input.shape}")
    print(f"测试输入dtype: {test_input.dtype}")
    try:
        test_output = net(test_input)
        print(f"测试输出形状: {test_output.shape}")
        print("前向传播测试通过!")
    except Exception as e:
        print(f"前向传播测试失败: {e}")
        raise

print("=" * 50)

# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=config["train"]["lr"], momentum=0.9, weight_decay=5e-4)

# 训练
if __name__ == "__main__":
    best_acc = 0
    print("Start Training, %s !" % config["net"])
    
    for epoch in range(config["train"]["pre_epoch"], config["train"]["epoch"]):
        print('\nEpoch: %d' % (epoch + 1))
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        
        for i, data in enumerate(trainloader, 0):
            length = len(trainloader)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 打印第一个batch的详细信息
            if i == 0 and epoch == 0:
                print(f"训练输入形状: {inputs.shape}")
                print(f"训练输入dtype: {inputs.dtype}")
                print(f"训练输入设备: {inputs.device}")
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            
            if i % 100 == 0:
                print('[epoch:%d, iter:%d/%d] Loss: %.03f | Acc: %.3f%% ' %
                      (epoch + 1, (i + 1), length, sum_loss / (i + 1), 100. * correct / total))

        # 测试
        print("Waiting Test!")
        with torch.no_grad():
            correct = 0
            total = 0
            net.eval()
            for test_i, data in enumerate(testloader):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            acc = 100. * correct / total
            print('测试分类准确率为：%.3f%%' % acc)
            
            print('Saving model......')
            torch.save(net.state_dict(), '%s/net_%03d_%.2f.pth' % 
                      (config["train"]["out_model_path"], epoch + 1, acc))

            if acc > best_acc:
                best_acc = acc
                torch.save(net.state_dict(), '%s/best_net_%.2f.pth' % 
                          (config["train"]["out_model_path"], best_acc))
                print('Best model updated!')

    print("Training Finished, TotalEPOCH=%d, Best Acc=%.2f%%" % (config["train"]["epoch"], best_acc))