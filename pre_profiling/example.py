"""
使用示例：pre_profiling模块的各种使用场景
"""

import torch
import torch.nn as nn
from torchvision.models import inception_v3, resnet50
import sys
import os

# 直接导入（不需要作为包）
from layer_config_parser import LayerConfigParser, LayerInfo
from pre_profiling import run_pre_profiling, PreProfiler, ProfilingResult


def example_1_basic():
    """示例1: 最基础的使用方式"""
    print("=" * 60)
    print("示例1: 基础使用")
    print("=" * 60)
    
    # 准备模型
    model = inception_v3(aux_logits=False, init_weights=True)
    
    # 运行pre_profiling
    results = run_pre_profiling(
        model=model,
        layer_config_path='./layers.config',
        output_path='inception_v3_profiling.txt',
        device='npu:7',  # 可以用 'npu:0' 或 'cuda:0'
        num_runs=5,
        warmup=2
    )
    
    print(f"\n成功测量了 {len(results)} 个模块")
    # 打印前3个结果
    for i, (name, result) in enumerate(list(results.items())[:3]):
        print(f"  {name}: FWD={result.forward_time_mean:.2f}us")


def example_2_parser_only():
    """示例2: 仅测试layer.config解析"""
    print("\n" + "=" * 60)
    print("示例2: 仅解析layer.config")
    print("=" * 60)
    
    # 创建测试用的layer.config
    test_config = """Hook ID:0; Name:Conv2d (128,3,299,299)
Next Layers:
Next Layer 0 Hook ID:1; Name:BatchNorm2d (128,32,149,149)
Previous Layers:
Input Tensor: tensor0 Is weight (global)?: 0, Size in byte: 137322496, Range:0--137322496
Output Tensor: tensor1 Is weight (global)?: 0, Size in byte: 363741184, Range:137322496--501063680
______________________________________________________________________________
Hook ID:1; Name:BatchNorm2d (128,32,149,149)
Next Layers:
Previous Layers:
Previous Layer 0 Hook ID:0; Name:Conv2d (128,3,299,299)
Input Tensor: tensor1 Is weight (global)?: 0, Size in byte: 363741184, Range:137322496--501063680
______________________________________________________________________________"""
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.config', delete=False) as f:
        f.write(test_config)
        temp_path = f.name
    
    try:
        layers = LayerConfigParser.parse(temp_path)
        print(f"\n解析了 {len(layers)} 层:")
        for hook_id, info in sorted(layers.items()):
            print(f"  Hook {hook_id}: {info.layer_type}")
            print(f"    Shape: {info.shape_info}")
            print(f"    Inputs: {len(info.input_tensors)} tensors")
            print(f"    Prev: {info.prev_layers}, Next: {info.next_layers}")
    finally:
        os.unlink(temp_path)


def example_3_custom_model():
    """示例3: 自定义模型"""
    print("\n" + "=" * 60)
    print("示例3: 自定义模型")
    print("=" * 60)
    
    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            return x
    
    model = MyModel()
    
    # 手动创建layer.config
    config_content = """Hook ID:0; Name:Conv2d (2,3,32,32)
______________________________________________________________________________
Hook ID:1; Name:BatchNorm2d (2,64,32,32)
______________________________________________________________________________
Hook ID:2; Name:ReLU (2,64,32,32)
______________________________________________________________________________"""
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.config', delete=False) as f:
        f.write(config_content)
        temp_path = f.name
    
    try:
        results = run_pre_profiling(
            model=model,
            layer_config_path=temp_path,
            output_path='my_model_profiling.txt',
            device='cpu',
            num_runs=3,
            warmup=1
        )
        
        print(f"\n测量完成: {len(results)} 个模块")
        for name, r in results.items():
            print(f"  {r.to_line()}")
    finally:
        os.unlink(temp_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--example', type=str, default='1', choices=['1','2','3','all'])
    args = parser.parse_args()
    
    if args.example == '1':
        example_1_basic()
    elif args.example == '2':
        example_2_parser_only()
    elif args.example == '3':
        example_3_custom_model()
    elif args.example == 'all':
        example_2_parser_only()
        example_3_custom_model()