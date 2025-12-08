#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Created on 2020/11/3 13:38
# Project:
# @Author: CaoYugang
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

# Auto transfer to NPU
import torch_npu
from torch_npu.contrib import transfer_to_npu


with open('./config.yaml', 'r', encoding='utf-8') as f_config:
    config = yaml.load(f_config.read(), Loader=yaml.FullLoader)

device = torch.device("npu:0" if torch.npu.is_available() else "cpu")

# ---------- 1. 数据：只取 1 张图 ----------
transform = transforms.Compose([
    transforms.Resize((config["width"], config["height"])),
    transforms.ToTensor(),
])
trainset = torchvision.datasets.CIFAR10(root=config["train"]["train_data"],
                                        train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=2, shuffle=False, num_workers=0)

# ---------- 2. 模型 ----------
net = inception_v3(num_classes=10, aux_logits=False).to(device)

# ---------- 3. 强制替换所有 F.relu -> nn.ReLU(inplace=False) ----------
def replace_all_relu(module, name=''):
    for child_name, child in list(module.named_children()):
        full_name = (name + '.' + child_name) if name else child_name
        if isinstance(child, nn.Sequential):
            new_layers = []
            for k, sub in child.named_children():
                new_layers.append(sub)
                if 'relu' in k.lower() and not isinstance(sub, (nn.ReLU, nn.ReLU6)):
                    new_layers.append(nn.ReLU(inplace=False))
            setattr(module, child_name, nn.Sequential(*new_layers))
        else:
            replace_all_relu(child, full_name)

replace_all_relu(net)

# ---------- 4. 注册 ReLU hook ----------
relu_count = 0
def relu_hook(m, inp, out):
    global relu_count
    relu_count += 1
    print(f'[ReLU hook {relu_count:02d}] {m} -> shape={out.shape}  val={out.flatten()[:5].tolist()}')

for m in net.modules():
    if isinstance(m, nn.ReLU):
        m.register_forward_hook(relu_hook)

# ---------- 5. 损失 & 优化 ----------
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

# ---------- 6. 单 batch 单元测试 ----------
net.train()
inputs, labels = next(iter(trainloader))
inputs, labels = inputs.to(device), labels.to(device)

optimizer.zero_grad()
outputs = net(inputs)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()

print('\n==== 单 batch 结束，共抓到', relu_count, '个 ReLU ====')