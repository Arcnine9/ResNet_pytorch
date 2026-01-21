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

from swap_manager import swapManager as swap_manager_mod
from swap_manager import hook, module_transfer

import torch_npu
from torch_npu.contrib import transfer_to_npu


with open('./config.yaml', 'r', encoding='utf-8') as f_config:
    config = yaml.load(f_config.read(), Loader=yaml.FullLoader)

device = torch.device(
    "npu:1" if torch.npu.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

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
    net = inception_v3(num_classes=len(classes), aux_logits=False)
    net = module_transfer.replace_functional(net, verbose=False)
else:
    raise Exception("Unknown network")

net = net.to(device)


# ================= Swap / Hook =================
swap_manager = swap_manager_mod.SwapManager()   # ★ 不再覆盖模块名
hook_manager = hook.HookManager(
    swap_manager,
    "prefetch.config",
    net
)


# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(
#     net.parameters(),
#     lr=config["train"]["lr"],
#     momentum=0.9,
#     weight_decay=5e-4
# )

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)


# ================= Training =================
if __name__ == "__main__":
    print("Start Training:", config["net"])

    for epoch in range(config["train"]["pre_epoch"], config["train"]["epoch"]):
        print(f"\nEpoch: {epoch + 1}")
        net.train()

        for i, (inputs, labels) in enumerate(trainloader):

            # ★ 再 reset issued_time
            torch.npu.synchronize()
            hook_manager.reset_issued_time()
            torch.npu.synchronize()
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # ★ 保证上一个 batch 的 swap 已完全结束
            torch.npu.synchronize()
            # swap_manager.clear()
            torch.npu.synchronize()
            # # 打印
            _, predicted = torch.max(outputs.data, 1)
            torch.npu.synchronize()
            acc = (predicted == labels).float().mean() * 100
            torch.npu.synchronize()
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

    # hook_manager.remove_hooks()
    print("Training Finished")
