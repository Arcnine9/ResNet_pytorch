#!/usr/bin/env python
# -*- encoding:utf-8 -*-

import os
import torch
import torch.nn as nn
from torchvision.models import resnet as ResNet
from torchvision.models import inception_v3
from swap_manager import hook, module_transfer   # 你的 FX 替换 + Hook 管理
import torch_npu
from torch_npu.contrib import transfer_to_npu   # 自动把 cuda 接口 remap

# ---------- 设备 ----------
device = torch.device("npu:1" if torch.npu.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

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
BATCH_SIZE = 4
NUM_BATCH  = 1
hook_manager = hook.register_all_hooks(net)

for batch_idx in range(NUM_BATCH):
    # ===== 每 batch 重置执行序号 =====
    # hook._LAYER_ID = 0          # 全局序号归零
    hook_manager.reset_issued_time()
    print(f"\n========== BATCH {batch_idx + 1} ==========")

    # 随机输入 & 标签
    inputs = torch.randn(BATCH_SIZE, 3, 224, 224, device=device)
    labels = torch.randint(0, 10, (BATCH_SIZE,), device=device)

    # ===== 前向 =====
    net.train()
    outputs = net(inputs)
    loss = criterion(outputs, labels)

    # ===== 反向 =====
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()




hook_manager.remove_hooks()


print("✅ ")