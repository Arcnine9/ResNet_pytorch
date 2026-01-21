#!/usr/bin/env python
# -*- encoding:utf-8 -*-

import os
import torch
import torch.nn as nn
from torchvision.models import resnet as ResNet
from torchvision.models import inception_v3
from swap_manager import swapManager as swap_manager_mod
from swap_manager import hook, module_transfer
import torch_npu
from torch_npu.contrib import transfer_to_npu   # 自动把 cuda 接口 remap

# ---------- 设备 ----------
device = torch.device("npu:0" if torch.npu.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# ---------- 模型 ----------
net = inception_v3(aux_logits=False, init_weights=True)
# net = ResNet.resnet152(num_classes=10)
# example = torch.randn(1, 3, 224, 224)          # 任意尺寸
net = module_transfer.replace_functional(net, verbose=False)
net = net.to(device)

# ---------- 损失 & 优化 ----------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

# ---------- 只跑 2 个 batch ----------
BATCH_SIZE = 128        # 配置中的 batch_size
NUM_BATCH = 20           # 模拟前两个 batch
IMG_HEIGHT = 75         # 配置中的 height
IMG_WIDTH = 75          # 配置中的 width
NUM_CLASSES = 1000      # InceptionV3 默认类数，和真实数据保持一致

swap_manager = swap_manager_mod.SwapManager()   # ★ 不再覆盖模块名
hook_manager = hook.HookManager(
    swap_manager,
    "prefetch.config",
    net
)

inputs = torch.randn(BATCH_SIZE, 3, IMG_HEIGHT, IMG_WIDTH, device=device)
labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), device=device)

for batch_idx in range(NUM_BATCH):
    # ===== 每 batch 重置执行序号 =====
    # hook._LAYER_ID = 0          # 全局序号归零

    # ★ 再 reset issued_time
    hook_manager.reset_issued_time()

    print(f"\n========== BATCH {batch_idx + 1} ==========")

    # 随机输入 & 标签

    inputs = inputs.to(device)
    labels = labels.to(device)
    # ===== 前向 =====
    net.train()
    outputs = net(inputs)
    loss = criterion(outputs, labels)

    # ===== 反向 =====
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # ★ 保证上一个 batch 的 swap 已完全结束
    torch_npu.npu.current_stream().synchronize()
    swap_manager.clear()
    # 打印
    _, predicted = torch.max(outputs.data, 1)
    acc = (predicted == labels).float().mean() * 100

    print(
        f"Loss: {loss.item():.4f} | Acc: {acc:.2f}%"
    )



hook_manager.remove_hooks()


print("✅ ")