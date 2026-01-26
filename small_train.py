#!/usr/bin/env python
# -*- encoding:utf-8 -*-

import os
import torch
import torch.nn as nn
from torchvision.models import resnet as ResNet
from torchvision.models import inception_v3

# 修正导入路径
from swap_manager import swap_manager as swap_manager_mod
from swap_manager import hook, module_transfer

import torch_npu
from torch_npu.contrib import transfer_to_npu  # 自动把 cuda 接口 remap

# ---------- 设备 ----------
device = torch.device("npu:0" if torch.npu.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
device_idx = device.index if device.type == "npu" else 0  # 提取设备索引

print(f"[INIT] Using device: {device}, device_idx: {device_idx}")

# 设置默认设备（关键！）
if device.type == "npu":
    torch_npu.npu.set_device(device_idx)

# ---------- 模型 ----------
net = inception_v3(aux_logits=False, init_weights=True)
# net = ResNet.resnet152(num_classes=10)
net = module_transfer.replace_functional(net, verbose=False)
net = net.to(device)

# 验证模型设备
sample_param = next(net.parameters())
print(f"[INIT] Model on device: {sample_param.device}")

# ---------- 损失 & 优化 ----------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

# ---------- 配置 ----------
BATCH_SIZE = 128
NUM_BATCH = 20
IMG_HEIGHT = 299
IMG_WIDTH = 299
NUM_CLASSES = 1000

# ---------- 创建 SwapManager 和 HookManager ----------
# 使用简化后的 SwapManager，只传 device
swap_manager = swap_manager_mod.SwapManager(device=device_idx)
hook_manager = hook.HookManager(
    swap_manager,
    "prefetch.config",
    net,
    device=device_idx
)

# ---------- 测试数据 ----------
inputs = torch.randn(BATCH_SIZE, 3, IMG_HEIGHT, IMG_WIDTH, device=device)
labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), device=device)

print(f"[INIT] Input device: {inputs.device}, shape: {inputs.shape}")

# ---------- 训练循环 ----------
for batch_idx in range(NUM_BATCH):
    hook_manager.reset_issued_time()
    print(f"\n========== BATCH {batch_idx + 1} ==========")

    # 前向传播
    net.train()
    outputs = net(inputs)
    loss = criterion(outputs, labels)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 同步当前 stream
    torch_npu.npu.current_stream().synchronize()
    
    # 清理本 batch 的 swap tensor
    swap_manager.clear()

    # 打印统计
    _, predicted = torch.max(outputs.data, 1)
    acc = (predicted == labels).float().mean() * 100
    print(f"Loss: {loss.item():.4f} | Acc: {acc:.2f}%")

# 清理
hook_manager.remove_hooks()
print("✅ Test completed")