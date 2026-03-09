#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import signal
import sys

# =========================
# 信号处理：确保 Ctrl+C 能终止程序
# =========================
def signal_handler(signum, frame):
    print(f"\n[Signal] 捕获信号 {signum}，正在清理并退出...")
    # 强制清理 NPU 上下文
    try:
        import torch_npu
        torch_npu.npu.synchronize()  # 尝试同步
        torch_npu.npu.empty_cache()
    except:
        pass
    sys.exit(1)

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # kill -15

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

from swap_manager import swapManager as swap_manager_mod
from swap_manager import hook, module_transfer

import torch_npu


with open('./config.yaml', 'r', encoding='utf-8') as f_config:
    config = yaml.load(f_config.read(), Loader=yaml.FullLoader)

# =========================
# ★ 关键修改：显式设置 device 并确保上下文正确
# =========================
if torch.npu.is_available():
    device_id = 7  # 或者从配置读取：config.get("device_id", 0)
    torch.npu.set_device(device_id)  # 立即设置当前线程的默认 device
    device = torch.device(f"npu:{device_id}")
else:
    device = torch.device("cpu")

print(f"[INIT] Using device: {device}")

if not os.path.exists(config["train"]["out_model_path"]):
    raise Exception("模型保存路径不存在")


# ================= Dataset =================
class CNNNetworkDataset(Dataset):
    def __init__(self, base_dataset, transform=None, should_invert=True):
        self.base_dataset = base_dataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img, label = self.base_dataset[index]
        if self.should_invert:
            img = PIL.ImageOps.invert(img)
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.base_dataset)


transform_train = transforms.Compose([
    transforms.Resize((config["width"], config["height"])),
    transforms.RandomHorizontalFlip(0.5 if config["train"]["rotating"] else 0),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Resize((config["width"], config["height"])),
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.CIFAR10(
    root=config["train"]["train_data"],
    train=True,
    download=True,
    transform=transform_train
)

test_dataset = torchvision.datasets.CIFAR10(
    root=config["train"]["test_data"],
    train=False,
    download=True,
    transform=transform_test
)

trainloader = DataLoader(
    train_dataset,
    batch_size=config["train"]["batch_size"],
    shuffle=True,
    num_workers=config["train"]["num_workers"]
)

testloader = DataLoader(
    test_dataset,
    batch_size=config["train"]["batch_size"],
    shuffle=True,
    num_workers=config["train"]["num_workers"]
)

classes = train_dataset.classes


# ================= Model =================
if config["net"] == "ResNet18":
    net = ResNet.resnet18(num_classes=len(classes))
elif config["net"] == "ResNet34":
    net = ResNet.resnet34(num_classes=len(classes))
elif config["net"] == "ResNet50":
    net = ResNet.resnet50(num_classes=len(classes))
elif config["net"] == "ResNet101":
    net = ResNet.resnet101(num_classes=len(classes))
elif config["net"] == "ResNet152":
    net = ResNet.resnet152(num_classes=len(classes))
elif config["net"] == "InceptionV3":
    net = inception_v3(aux_logits=False, init_weights=True)
    net = module_transfer.replace_functional(net, verbose=False)
else:
    raise Exception("Unknown network")

# ★ 确保模型在正确的 device 上
net = net.to(device)

hook_verbose = config["train"].get("hook_verbose", False)

# ================= Swap / Hook =================
enable_vector_transfer = config["train"].get("enable_vector_transfer", True)

# ★ 关键修改：显式传递 device 给 SwapManager
if enable_vector_transfer:
    # 确保在正确的 device 上下文中创建 SwapManager
    with torch.npu.device(device.index):
        swap_manager = swap_manager_mod.SwapManager(device=device.index)
    
    hook_manager = hook.HookManager(
        swap_manager,
        "prefetch.config",
        net,
        verbose=hook_verbose
    )
    print(f"向量迁移功能已启用 (enable_vector_transfer={enable_vector_transfer}, device={device})")
else:
    swap_manager = None
    hook_manager = None
    print(f"向量迁移功能已禁用 (enable_vector_transfer={enable_vector_transfer})")


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)


# ================= Training =================
if __name__ == "__main__":
    print("Start Training:", config["net"])

    for epoch in range(config["train"]["pre_epoch"], config["train"]["epoch"]):
        print(f"\nEpoch: {epoch + 1}")
        net.train()

        for i, (inputs, labels) in enumerate(trainloader):

            if hook_manager is not None:
                hook_manager.reset_issued_time()

            # ★ 确保数据在正确的 device 上
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # ★ 关键修改：确保 clear 在正确的 device 上下文中执行
            if swap_manager is not None:
                with torch.npu.device(device.index):
                    swap_manager.clear()
            
            # 打印
            _, predicted = torch.max(outputs.data, 1)
            acc = (predicted == labels).float().mean() * 100
            print(
                f"[epoch:{epoch+1}, iter:{i+1}/{len(trainloader)}] "
                f"Loss: {loss.item():.4f} | Acc: {acc:.2f}%"
            )

        # ===== test =====
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in testloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100. * correct / total
        print(f"Test Acc: {acc:.3f}%")

        torch.save(
            net.state_dict(),
            f"{config['train']['out_model_path']}/net_{epoch+1}_{acc:.3f}.pth"
        )

    if hook_manager is not None:
        hook_manager.remove_hooks()
    
    print("Training Finished")