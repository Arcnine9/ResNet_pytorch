#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import sys
import time
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


# ================= 低开销 NPU 显存可视化工具 =================
class NPUMemoryVisualizer:
    """
    极简 NPU 显存监控：只在关键点位采样，可视化显示
    开销极低：每次采样 < 1ms，不开启时不产生任何开销
    """
    
    # 显存条颜色定义
    COLORS = {
        'green': '\033[92m',    # 正常
        'yellow': '\033[93m',   # 警告 (>70%)
        'red': '\033[91m',      # 危险 (>85%)
        'blue': '\033[94m',     # 信息
        'reset': '\033[0m',
        'bold': '\033[1m'
    }
    
    def __init__(self, enabled=True, bar_length=30):
        self.enabled = enabled and torch.npu.is_available()
        self.bar_length = bar_length
        self.max_seen = 0  # 观察到的峰值
        self.samples = []  # 可选：记录关键点位数据
        
        if self.enabled:
            # 预热：获取总显存
            self.total_mem = self._get_total_memory()
            print(f"[Memory] NPU 总显存: {self.total_mem:.0f} MiB "
                  f"({self.total_mem/1024:.2f} GiB)")
    
    def _get_memory_stats(self):
        """获取当前显存统计（单位：MiB）"""
        if not self.enabled:
            return None
        
        # 方法1：torch_npu 原生接口（推荐，开销最低）
        try:
            allocated = torch.npu.memory_allocated() / 1024**2
            reserved = torch.npu.memory_reserved() / 1024**2
            # 获取实际占用（HBM）
            if hasattr(torch_npu, '_npu_memory_stats'):
                stats = torch_npu._npu_memory_stats()
                hbm_used = stats.get('allocated_bytes', allocated * 1024**2) / 1024**2
            else:
                hbm_used = allocated
            return {
                'allocated': allocated,
                'reserved': reserved,
                'hbm_used': hbm_used
            }
        except:
            return {'allocated': 0, 'reserved': 0, 'hbm_used': 0}
    
    def _get_total_memory(self):
        """获取 NPU 总显存"""
        try:
            # 通过设备属性获取
            device_props = torch.npu.get_device_properties(0)
            return device_props.total_memory / 1024**2  # MiB
        except:
            # 备用：使用 npu-smi 一次获取
            import subprocess
            result = subprocess.run(['npu-smi', 'info', '-t', 'memory'], 
                                  capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if line.strip() and line[0].isdigit():
                    parts = line.split()
                    return int(parts[5].replace('MiB', ''))
            return 32768  # 默认值 32GB
    
    def sample(self, tag=""):
        """
        采样并可视化显示显存占用
        只在关键点位调用，如：batch开始、forward后、clear后
        """
        if not self.enabled:
            return
        
        stats = self._get_memory_stats()
        if stats is None:
            return
        
        current = stats['hbm_used']
        self.max_seen = max(self.max_seen, current)
        pct = (current / self.total_mem) * 100
        
        # 记录关键点位（可选）
        self.samples.append({
            'tag': tag,
            'time': time.time(),
            'used': current,
            'pct': pct
        })
        
        # 生成可视化进度条
        filled = int(self.bar_length * pct / 100)
        bar = '█' * filled + '░' * (self.bar_length - filled)
        
        # 颜色选择
        if pct > 85:
            color = self.COLORS['red']
        elif pct > 70:
            color = self.COLORS['yellow']
        else:
            color = self.COLORS['green']
        
        # 格式化输出：标签 + 进度条 + 数值
        tag_str = f"[{tag:12s}]" if tag else "[memory]"
        print(f"\r{self.COLORS['bold']}{tag_str}{self.COLORS['reset']} "
              f"{color}|{bar}|{self.COLORS['reset']} "
              f"{current:6.0f}/{self.total_mem:.0f} MiB "
              f"({pct:5.1f}%) "
              f"[峰值:{self.max_seen:.0f}]", end='', flush=True)
        
        # 如果是换行标签，打印换行
        if tag in ['batch_end', 'epoch_end', 'summary']:
            print()  # 换行
    
    def show_delta(self, tag="", prev_stats=None):
        """显示显存变化（用于对比优化前后）"""
        if not self.enabled:
            return None
        
        current = self._get_memory_stats()
        self.sample(tag)
        
        if prev_stats:
            delta = current['hbm_used'] - prev_stats['hbm_used']
            sign = '+' if delta > 0 else ''
            print(f" Δ{sign}{delta:.0f}MiB", end='', flush=True)
        
        return current
    
    def summary(self):
        """显示统计摘要"""
        if not self.enabled or not self.samples:
            return
        
        print(f"\n{self.COLORS['bold']}========== 显存监控摘要 =========={self.COLORS['reset']}")
        print(f"总显存: {self.total_mem:.0f} MiB ({self.total_mem/1024:.2f} GiB)")
        print(f"峰值占用: {self.max_seen:.0f} MiB ({self.max_seen/self.total_mem*100:.1f}%)")
        
        if len(self.samples) > 0:
            avg = sum(s['used'] for s in self.samples) / len(self.samples)
            print(f"平均占用: {avg:.0f} MiB ({avg/self.total_mem*100:.1f}%)")
            print(f"采样点数: {len(self.samples)}")
        
        # 显示关键点位对比（如果有 clear 操作）
        clears = [s for s in self.samples if 'clear' in s['tag']]
        if clears:
            print(f"\n显存清理效果:")
            for i, s in enumerate(clears[:3]):  # 显示前3次
                print(f"  清理 #{i+1}: {s['used']:.0f} MiB @ {s['tag']}")
        
        print("=" * 40)


# ================= 配置加载 =================
with open('./config.yaml', 'r', encoding='utf-8') as f_config:
    config = yaml.load(f_config.read(), Loader=yaml.FullLoader)

device = torch.device(
    "npu:0" if torch.npu.is_available()
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
    net = inception_v3(aux_logits=False, init_weights=True)
    net = module_transfer.replace_functional(net, verbose=False)
else:
    raise Exception("Unknown network")

net = net.to(device)

hook_verbose = config["train"].get("hook_verbose", False)

# ================= Swap / Hook =================
swap_manager = swap_manager_mod.SwapManager()
hook_manager = hook.HookManager(
    swap_manager,
    "prefetch.config",
    net,
    verbose=hook_verbose
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)


# ================= Training =================
if __name__ == "__main__":
    print("Start Training:", config["net"])
    
    # 初始化显存可视化器（通过环境变量控制是否启用）
    # export NPU_MEM_VIS=1 启用，不设置或0则禁用
    mem_vis = NPUMemoryVisualizer(enabled=os.getenv('NPU_MEM_VIS', '1') == '1')
    
    # 初始显存状态
    mem_vis.sample("init")

    for epoch in range(config["train"]["pre_epoch"], config["train"]["epoch"]):
        print(f"\nEpoch: {epoch + 1}")
        net.train()

        for i, (inputs, labels) in enumerate(trainloader):
            
            # 可选：每N个batch显示一次，避免刷屏
            show_mem = (i % 10 == 0) or (i < 3)  # 前3个和每10个显示
            
            if show_mem:
                mem_vis.sample(f"batch{i}_start")

            hook_manager.reset_issued_time()

            inputs = inputs.to(device)
            labels = labels.to(device)
            
            if show_mem:
                mem_vis.sample(f"batch{i}_data")

            optimizer.zero_grad()

            outputs = net(inputs)
            
            if show_mem:
                mem_vis.sample(f"batch{i}_fwd")

            loss = criterion(outputs, labels)
            loss.backward()
            
            if show_mem:
                mem_vis.sample(f"batch{i}_bwd")

            optimizer.step()

            if show_mem:
                mem_vis.sample(f"batch{i}_step")

            # 关键：显示 swap_manager.clear() 的效果！
            swap_manager.clear()
            
            if show_mem:
                # 重点显示清理后的显存变化
                mem_vis.sample(f"batch{i}_clear")
                print()  # 换行，完成这个batch的显示

            # 打印训练指标（在同一行或下一行）
            _, predicted = torch.max(outputs.data, 1)
            acc = (predicted == labels).float().mean() * 100
            print(
                f"[epoch:{epoch+1}, iter:{i+1}/{len(trainloader)}] "
                f"Loss: {loss.item():.4f} | Acc: {acc:.2f}%"
            )

        # Epoch 结束显示摘要
        mem_vis.sample("epoch_end")

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

    # 训练结束，显示完整摘要
    mem_vis.sample("training_end")
    mem_vis.summary()
    
    print("Training Finished")