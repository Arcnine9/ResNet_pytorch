#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import sys
import signal

# 添加父目录到路径，以便导入 swap_manager
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torchvision.models import inception_v3

# 必须先导入 torch_npu
import torch_npu

# 导入模块（修正路径）
try:
    from pre_profiling import run_pre_profiling, PreProfiler, LayerConfigParser
    from swap_manager.module_transfer import replace_functional  # 上级目录的 swap_manager
except ImportError as e:
    print(f"[ERROR] 导入失败: {e}")
    print("请确保目录结构正确：")
    print("  ResNet_pytorch/")
    print("    ├── pre_profiling/")
    print("    │   ├── pre_profiling.py")
    print("    │   └── example.py  (当前文件)")
    print("    └── swap_manager/")
    print("        └── module_transfer.py")
    sys.exit(1)


# =========================
# 信号处理（同你的 train.py）
# =========================
def signal_handler(signum, frame):
    print(f"\n[Signal] 捕获信号 {signum}，正在清理...")
    try:
        torch_npu.npu.synchronize()
        torch_npu.npu.empty_cache()
    except:
        pass
    sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# =========================
# 示例 1: 最简使用（InceptionV3，默认）
# =========================
def example_basic():
    """最简使用：InceptionV3（与你的 train.py 一致）"""
    print("=" * 60)
    print("示例 1: InceptionV3 最简测量")
    print("=" * 60)
    
    # 创建 InceptionV3（与你的 train.py 完全一致）
    model = inception_v3(aux_logits=False, init_weights=True)
    
    # 使用 FX 转换（从你的 swap_manager 导入）
    model = replace_functional(model, verbose=False)
    
    # 检查 layer.config
    if not os.path.exists('./layers.config'):
        print("[INFO] 未找到 layers.config，创建 InceptionV3 测试配置...")
        create_inceptionv3_test_config()
    
    # 一行代码完成测量（默认 npu:7）
    results = run_pre_profiling(
        model=model,
        layer_config_path='./layers.config',
        output_path='inceptionv3_profiling.txt',
        device_id=7,        # 同你的 train.py
        num_runs=5,         # 运行 5 次取平均
        warmup=2            # 预热 2 次
    )
    
    print(f"\n✓ 测量完成！共 {len(results)} 个模块")
    print("\n前 5 个模块结果：")
    for i, (name, r) in enumerate(list(results.items())[:5]):
        total = r.forward_time_mean + r.backward_time_mean
        print(f"  {name:30s} FWD={r.forward_time_mean:8.2f}us  BWD={r.backward_time_mean:8.2f}us  Total={total:8.2f}us")


# =========================
# 示例 2: 详细控制（查看所有层信息）
# =========================
def example_detailed():
    """详细控制：查看 InceptionV3 所有层"""
    print("\n" + "=" * 60)
    print("示例 2: InceptionV3 详细分析")
    print("=" * 60)
    
    model = inception_v3(aux_logits=False, init_weights=True)
    model = replace_functional(model, verbose=True)  # 开启 verbose 查看转换
    
    if not os.path.exists('./layers.config'):
        create_inceptionv3_test_config()
    
    # 使用 PreProfiler 类
    profiler = PreProfiler(model, './layers.config', device_id=7)
    
    # 先只解析，看看匹配了多少层
    print(f"\n模型结构摘要：")
    leaf_modules = [name for name, m in model.named_modules() 
                   if len(list(m.children())) == 0 or isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU))]
    print(f"  叶子模块数量: {len(leaf_modules)}")
    print(f"  layer.config 层数: {len(profiler.layer_info_dict)}")
    
    # 运行测量
    results = profiler.run(num_runs=3, warmup=1, measure_backward=True)
    
    # 摘要统计
    summary = profiler.get_results_summary()
    print(f"\n{'='*60}")
    print(f"总模块数: {summary['num_modules']}")
    print(f"总前向时间: {summary['total_forward_time_us']/1000:.2f} ms")
    print(f"总反向时间: {summary['total_backward_time_us']/1000:.2f} ms")
    print(f"总时间: {summary['total_time_us']/1000:.2f} ms")
    
    # 找出最慢的 3 个模块
    sorted_results = sorted(results.items(), 
                          key=lambda x: x[1].forward_time_mean + x[1].backward_time_mean,
                          reverse=True)
    print(f"\n最慢的 3 个模块：")
    for name, r in sorted_results[:3]:
        total = r.forward_time_mean + r.backward_time_mean
        print(f"  1. {name}: {total:.2f} us (FWD={r.forward_time_mean:.2f}, BWD={r.backward_time_mean:.2f})")
    
    # 保存 JSON
    profiler.save_results('inceptionv3_detailed.json', format='json')


# =========================
# 示例 3: 仅解析 layer.config
# =========================
def example_parse_only():
    """仅解析 layer.config，查看 InceptionV3 结构"""
    print("\n" + "=" * 60)
    print("示例 3: 仅解析 InceptionV3 layer.config")
    print("=" * 60)
    
    if not os.path.exists('./layers.config'):
        create_inceptionv3_test_config()
    
    layers = LayerConfigParser.parse('./layers.config')
    
    print(f"\nInceptionV3 结构解析：共 {len(layers)} 个层")
    print(f"\n{'Hook ID':<8} {'Layer Type':<20} {'Shape':<30} {'Params':<10}")
    print("-" * 80)
    
    for hook_id in sorted(layers.keys())[:10]:
        info = layers[hook_id]
        params = sum(t.get('size', 0) for t in info.weight_tensors) / 4
        print(f"{hook_id:<8} {info.layer_type:<20} {str(info.shape_info):<30} {int(params):<10}")
    
    if len(layers) > 10:
        print(f"... 还有 {len(layers)-10} 个层 ...")


# =========================
# 辅助函数：创建 InceptionV3 测试配置
# =========================
def create_inceptionv3_test_config():
    """创建 InceptionV3 风格的测试 layer.config"""
    test_config = """Hook ID:0; Name:Conv2d (128,3,299,299)
Next Layers:
Next Layer 0 Hook ID:0; Name:BatchNorm2d (128,32,149,149)
Previous Layers:
Input Tensor: tensor0 Is weight (global)?: 0, Size in byte: 137322496, Range:0--137322496
Output Tensor: tensor1 Is weight (global)?: 0, Size in byte: 363741184, Range:137322496--501063680
Weight Tensor: tensor2 Is weight (global)?: 1, Size in byte: 4096, Range:0--4096
______________________________________________________________________________
Hook ID:1; Name:BatchNorm2d (128,32,149,149)
Next Layers:
Next Layer 0 Hook ID:0; Name:ReLU (128,32,149,149)
Previous Layers:
Previous Layer 0 Hook ID:0; Name:Conv2d (128,3,299,299)
Input Tensor: tensor1 Is weight (global)?: 0, Size in byte: 363741184, Range:137322496--501063680
Output Tensor: tensor5 Is weight (global)?: 0, Size in byte: 363741184, Range:638390272--1002131456
______________________________________________________________________________
Hook ID:2; Name:ReLU (128,32,149,149)
Next Layers:
Next Layer 0 Hook ID:0; Name:Conv2d (128,32,149,149)
Previous Layers:
Previous Layer 0 Hook ID:0; Name:BatchNorm2d (128,32,149,149)
Input Tensor: tensor5 Is weight (global)?: 0, Size in byte: 363741184, Range:638390272--1002131456
Output Tensor: tensor5 Is weight (global)?: 0, Size in byte: 363741184, Range:638390272--1002131456
______________________________________________________________________________
Hook ID:3; Name:Conv2d (128,32,149,149)
Next Layers:
Next Layer 0 Hook ID:0; Name:BatchNorm2d (128,32,147,147)
Previous Layers:
Previous Layer 0 Hook ID:0; Name:ReLU (128,32,149,149)
Input Tensor: tensor5 Is weight (global)?: 0, Size in byte: 363741184, Range:638390272--1002131456
Output Tensor: tensor20 Is weight (global)?: 0, Size in byte: 354041856, Range:2457124864--2811166720
______________________________________________________________________________
Hook ID:4; Name:BatchNorm2d (128,32,147,147)
Next Layers:
Next Layer 0 Hook ID:0; Name:ReLU (128,32,147,147)
Previous Layers:
Previous Layer 0 Hook ID:0; Name:Conv2d (128,32,149,149)
Input Tensor: tensor20 Is weight (global)?: 0, Size in byte: 354041856, Range:2457124864--2811166720
Output Tensor: tensor24 Is weight (global)?: 0, Size in byte: 354041856, Range:3174944768--3528986624
______________________________________________________________________________
Hook ID:5; Name:ReLU (128,32,147,147)
Next Layers:
Previous Layers:
Previous Layer 0 Hook ID:0; Name:BatchNorm2d (128,32,147,147)
Input Tensor: tensor24 Is weight (global)?: 0, Size in byte: 354041856, Range:3174944768--3528986624
Output Tensor: tensor24 Is weight (global)?: 0, Size in byte: 354041856, Range:3174944768--3528986624
______________________________________________________________________________"""
    
    with open('./layers.config', 'w') as f:
        f.write(test_config)
    print("[INFO] 已创建 InceptionV3 测试用的 layers.config（6个层）")


# =========================
# 主函数
# =========================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Pre-profiling for InceptionV3')
    parser.add_argument('--example', type=int, default=1, 
                       choices=[1, 2, 3, 0],
                       help='选择示例: 1=基础, 2=详细, 3=仅解析, 0=全部')
    
    args = parser.parse_args()
    
    # 检查 NPU
    if not torch.npu.is_available():
        print("[ERROR] NPU 不可用，请检查 torch_npu 安装")
        sys.exit(1)
    
    print(f"[INFO] NPU 可用，使用设备: npu:7 (默认)")
    print(f"[INFO] 当前目录: {os.getcwd()}")
    
    try:
        if args.example == 0:
            example_basic()
            example_detailed()
            example_parse_only()
        elif args.example == 1:
            example_basic()
        elif args.example == 2:
            example_detailed()
        elif args.example == 3:
            example_parse_only()
            
        print("\n" + "=" * 60)
        print("示例运行完成！输出文件：")
        if os.path.exists('inceptionv3_profiling.txt'):
            print("  - inceptionv3_profiling.txt (基础结果)")
        if os.path.exists('inceptionv3_detailed.json'):
            print("  - inceptionv3_detailed.json (详细JSON)")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] 运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # 清理
        try:
            torch_npu.npu.empty_cache()
        except:
            pass