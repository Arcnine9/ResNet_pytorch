#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import sys
import signal

# 添加父目录到路径，以便导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torchvision.models import inception_v3, resnet50

# 必须先导入 torch_npu
import torch_npu

# 导入模块
try:
    from pre_profiling import run_pre_profiling, PreProfiler, LayerConfigParser
    from swap_manager.module_transfer import replace_functional  # 上级目录的 swap_manager
except ImportError as e:
    print(f"[ERROR] 导入失败: {e}")
    print("请确保目录结构正确：")
    print("  project/")
    print("    ├── pre_profiling.py")
    print("    ├── example.py  (当前文件)")
    print("    └── swap_manager/")
    print("        └── module_transfer.py")
    sys.exit(1)


# =========================
# 信号处理（同 train.py）
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
# 示例 1: 最简使用（自动探测映射）
# =========================
def example_basic():
    """最简使用：自动探测HookID到Module的映射"""
    print("=" * 60)
    print("示例 1: 自动探测映射并测量（InceptionV3）")
    print("=" * 60)
    
    # 创建 InceptionV3
    model = inception_v3(aux_logits=False, init_weights=True)
    
    # 可选：使用 FX 转换（如果你有swap_manager）
    try:
        model = replace_functional(model, verbose=False)
        print("[INFO] 已使用FX转换functional算子")
    except:
        pass
    
    # 检查 layer.config
    if not os.path.exists('./layers.config'):
        print("[INFO] 未找到 layers.config，创建测试配置...")
        create_test_config_for_inceptionv3()
    
    # 运行profiling（自动探测映射）
    results = run_pre_profiling(
        model=model,
        layer_config_path='./layers.config',
        output_path='inceptionv3_profiling.txt',
        device_id=7,
        num_runs=5,
        warmup=2,
        verbose=True  # 显示详细探测信息
    )
    
    print(f"\n✓ 测量完成！共 {len(results)} 个模块")


# =========================
# 示例 2: 仅探测映射关系（不测量）
# =========================
def example_probe_only():
    """仅探测HookID到Module的映射，不进行测量"""
    print("\n" + "=" * 60)
    print("示例 2: 仅探测HookID映射关系（小batch）")
    print("=" * 60)
    
    model = inception_v3(aux_logits=False, init_weights=True)
    
    if not os.path.exists('./layers.config'):
        create_test_config_for_inceptionv3()
    
    # 创建profiler并仅探测
    profiler = PreProfiler(model, './layers.config', device_id=7)
    
    print("[INFO] 开始探测（使用batch=1小输入）...")
    mapping = profiler.probe_module_mapping(verbose=True)
    
    print(f"\n探测到的映射关系（共{len(mapping)}个）：")
    print(f"{'Hook ID':<8} {'Module Name':<40} {'Layer Type':<15}")
    print("-" * 80)
    
    for hook_id in sorted(mapping.keys()):
        module_name = mapping[hook_id]
        layer_info = profiler.layer_info_dict.get(hook_id)
        layer_type = layer_info.layer_type if layer_info else "Unknown"
        print(f"{hook_id:<8} {module_name:<40} {layer_type:<15}")
    
    # 显示模型结构对比
    print(f"\n模型实际结构（前10个叶子模块）：")
    leaf_modules = [(n, m.__class__.__name__) for n, m in model.named_modules() 
                   if len(list(m.children())) == 0][:10]
    for name, type_name in leaf_modules:
        print(f"  {name:40s} {type_name}")


# =========================
# 示例 3: 详细控制（查看形状修复过程）
# =========================
def example_detailed():
    """详细控制：查看输入形状修复过程"""
    print("\n" + "=" * 60)
    print("示例 3: 详细测量（带形状修复日志）")
    print("=" * 60)
    
    model = resnet50(weights=None)
    
    if not os.path.exists('./layers.config'):
        create_test_config_for_resnet50()
    
    # 使用 PreProfiler 类
    profiler = PreProfiler(model, './layers.config', device_id=7)
    
    # 运行测量（verbose=True显示形状修复）
    results = profiler.run(num_runs=3, warmup=1, measure_backward=True, verbose=True)
    
    # 摘要统计
    summary = profiler.get_results_summary()
    if summary:
        print(f"\n{'='*60}")
        print(f"总模块数: {summary['num_modules']}")
        print(f"总前向时间: {summary['total_forward_time_us']/1000:.2f} ms")
        print(f"总反向时间: {summary['total_backward_time_us']/1000:.2f} ms")
        
        # 找出最慢的 3 个模块
        sorted_results = sorted(results.items(), 
                              key=lambda x: x[1].forward_time_mean + x[1].backward_time_mean,
                              reverse=True)
        print(f"\n最慢的 3 个模块：")
        for name, r in sorted_results[:3]:
            total = r.forward_time_mean + r.backward_time_mean
            print(f"  1. {name}: {total:.2f} us (FWD={r.forward_time_mean:.2f}, BWD={r.backward_time_mean:.2f})")
    
    # 保存 JSON
    profiler.save_results('resnet50_detailed.json', format='json')


# =========================
# 辅助函数：创建测试配置
# =========================
def create_test_config_for_inceptionv3():
    """创建InceptionV3风格的测试配置（前8层）"""
    config_content = """Hook ID:0; Name:Conv2d_1a_3x3.conv (2,3,299,299)
Next Layers:
Next Layer 0 Hook ID:1; Name:Conv2d_1a_3x3.bn (2,32,149,149)
Previous Layers:
Input Tensor: tensor0 Is weight (global)?: 0, Size in byte: 720756, Range:0--720756
Output Tensor: tensor1 Is weight (global)?: 0, Size in byte: 3845120, Range:720756--4565876
Weight Tensor: tensor2 Is weight (global)?: 1, Size in byte: 3456, Range:0--3456
______________________________________________________________________________
Hook ID:1; Name:Conv2d_1a_3x3.bn (2,32,149,149)
Next Layers:
Next Layer 0 Hook ID:2; Name:Conv2d_2a_3x3.conv (2,32,149,149)
Previous Layers:
Previous Layer 0 Hook ID:0; Name:Conv2d_1a_3x3.conv (2,3,299,299)
Input Tensor: tensor1 Is weight (global)?: 0, Size in byte: 3845120, Range:720756--4565876
Output Tensor: tensor3 Is weight (global)?: 0, Size in byte: 3845120, Range:5000000--8845120
Weight Tensor: tensor4 Is weight (global)?: 1, Size in byte: 128, Range:0--128
______________________________________________________________________________
Hook ID:2; Name:Conv2d_2a_3x3.conv (2,32,149,149)
Next Layers:
Next Layer 0 Hook ID:3; Name:Conv2d_2a_3x3.bn (2,32,147,147)
Previous Layers:
Previous Layer 0 Hook ID:1; Name:Conv2d_1a_3x3.bn (2,32,149,149)
Input Tensor: tensor3 Is weight (global)?: 0, Size in byte: 3845120, Range:5000000--8845120
Output Tensor: tensor5 Is weight (global)?: 0, Size in byte: 3763200, Range:10000000--13763200
Weight Tensor: tensor6 Is weight (global)?: 1, Size in byte: 36864, Range:0--36864
______________________________________________________________________________
Hook ID:3; Name:Conv2d_2a_3x3.bn (2,32,147,147)
Next Layers:
Next Layer 0 Hook ID:4; Name:Conv2d_2b_3x3.conv (2,32,147,147)
Previous Layers:
Previous Layer 0 Hook ID:2; Name:Conv2d_2a_3x3.conv (2,32,149,149)
Input Tensor: tensor5 Is weight (global)?: 0, Size in byte: 3763200, Range:10000000--13763200
Output Tensor: tensor7 Is weight (global)?: 0, Size in byte: 3763200, Range:15000000--18763200
Weight Tensor: tensor8 Is weight (global)?: 1, Size in byte: 128, Range:0--128
______________________________________________________________________________
Hook ID:4; Name:Conv2d_2b_3x3.conv (2,32,147,147)
Next Layers:
Next Layer 0 Hook ID:5; Name:Conv2d_2b_3x3.bn (2,64,147,147)
Previous Layers:
Previous Layer 0 Hook ID:3; Name:Conv2d_2a_3x3.bn (2,32,147,147)
Input Tensor: tensor7 Is weight (global)?: 0, Size in byte: 3763200, Range:15000000--18763200
Output Tensor: tensor9 Is weight (global)?: 0, Size in byte: 7526400, Range:20000000--27526400
Weight Tensor: tensor10 Is weight (global)?: 1, Size in byte: 73728, Range:0--73728
______________________________________________________________________________
Hook ID:5; Name:Conv2d_2b_3x3.bn (2,64,147,147)
Next Layers:
Next Layer 0 Hook ID:6; Name:maxpool1 (2,64,73,73)
Previous Layers:
Previous Layer 0 Hook ID:4; Name:Conv2d_2b_3x3.conv (2,32,147,147)
Input Tensor: tensor9 Is weight (global)?: 0, Size in byte: 7526400, Range:20000000--27526400
Output Tensor: tensor11 Is weight (global)?: 0, Size in byte: 7526400, Range:28000000--35526400
Weight Tensor: tensor12 Is weight (global)?: 1, Size in byte: 256, Range:0--256
______________________________________________________________________________
Hook ID:6; Name:maxpool1 (2,64,73,73)
Next Layers:
Next Layer 0 Hook ID:7; Name:Conv2d_3b_1x1.conv (2,64,73,73)
Previous Layers:
Previous Layer 0 Hook ID:5; Name:Conv2d_2b_3x3.bn (2,64,147,147)
Input Tensor: tensor11 Is weight (global)?: 0, Size in byte: 7526400, Range:28000000--35526400
Output Tensor: tensor13 Is weight (global)?: 0, Size in byte: 1881600, Range:36000000--37881600
______________________________________________________________________________
Hook ID:7; Name:Conv2d_3b_1x1.conv (2,64,73,73)
Next Layers:
Next Layer 0 Hook ID:8; Name:Conv2d_3b_1x1.bn (2,80,73,73)
Previous Layers:
Previous Layer 0 Hook ID:6; Name:maxpool1 (2,64,73,73)
Input Tensor: tensor13 Is weight (global)?: 0, Size in byte: 1881600, Range:36000000--37881600
Output Tensor: tensor14 Is weight (global)?: 0, Size in byte: 2352000, Range:38000000--40352000
Weight Tensor: tensor15 Is weight (global)?: 1, Size in byte: 5120, Range:0--5120
______________________________________________________________________________
Hook ID:8; Name:Conv2d_3b_1x1.bn (2,80,73,73)
Next Layers:
Next Layer 0 Hook ID:9; Name:Conv2d_4a_3x3.conv (2,80,71,71)
Previous Layers:
Previous Layer 0 Hook ID:7; Name:Conv2d_3b_1x1.conv (2,64,73,73)
Input Tensor: tensor14 Is weight (global)?: 0, Size in byte: 2352000, Range:38000000--40352000
Output Tensor: tensor16 Is weight (global)?: 0, Size in byte: 2352000, Range:41000000--43352000
Weight Tensor: tensor17 Is weight (global)?: 1, Size in byte: 320, Range:0--320
______________________________________________________________________________"""
    
    with open('./layers.config', 'w') as f:
        f.write(config_content)
    print("[INFO] 已创建InceptionV3测试配置（9层）")


def create_test_config_for_resnet50():
    """创建ResNet50风格的简单测试配置"""
    config_content = """Hook ID:0; Name:conv1 (2,3,224,224)
Next Layers:
Next Layer 0 Hook ID:1; Name:bn1 (2,64,112,112)
Previous Layers:
Input Tensor: tensor0 Is weight (global)?: 0, Size in byte: 602112, Range:0--602112
Output Tensor: tensor1 Is weight (global)?: 0, Size in byte: 1605632, Range:602112--2207744
Weight Tensor: tensor2 Is weight (global)?: 1, Size in byte: 9408, Range:0--9408
______________________________________________________________________________
Hook ID:1; Name:bn1 (2,64,112,112)
Next Layers:
Next Layer 0 Hook ID:2; Name:relu (2,64,112,112)
Previous Layers:
Previous Layer 0 Hook ID:0; Name:conv1 (2,3,224,224)
Input Tensor: tensor1 Is weight (global)?: 0, Size in byte: 1605632, Range:602112--2207744
Output Tensor: tensor3 Is weight (global)?: 0, Size in byte: 1605632, Range:3000000--4605632
Weight Tensor: tensor4 Is weight (global)?: 1, Size in byte: 256, Range:0--256
______________________________________________________________________________
Hook ID:2; Name:relu (2,64,112,112)
Next Layers:
Next Layer 0 Hook ID:3; Name:maxpool (2,64,112,112)
Previous Layers:
Previous Layer 0 Hook ID:1; Name:bn1 (2,64,112,112)
Input Tensor: tensor3 Is weight (global)?: 0, Size in byte: 1605632, Range:3000000--4605632
Output Tensor: tensor3 Is weight (global)?: 0, Size in byte: 1605632, Range:3000000--4605632
______________________________________________________________________________
Hook ID:3; Name:maxpool (2,64,56,56)
Next Layers:
Next Layer 0 Hook ID:4; Name:layer1.0.conv1 (2,64,56,56)
Previous Layers:
Previous Layer 0 Hook ID:2; Name:relu (2,64,112,112)
Input Tensor: tensor3 Is weight (global)?: 0, Size in byte: 1605632, Range:3000000--4605632
Output Tensor: tensor5 Is weight (global)?: 0, Size in byte: 401408, Range:5000000--5401408
______________________________________________________________________________
Hook ID:4; Name:layer1.0.conv1 (2,64,56,56)
Next Layers:
Next Layer 0 Hook ID:5; Name:layer1.0.bn1 (2,64,56,56)
Previous Layers:
Previous Layer 0 Hook ID:3; Name:maxpool (2,64,56,56)
Input Tensor: tensor5 Is weight (global)?: 0, Size in byte: 401408, Range:5000000--5401408
Output Tensor: tensor6 Is weight (global)?: 0, Size in byte: 401408, Range:6000000--6401408
Weight Tensor: tensor7 Is weight (global)?: 1, Size in byte: 4096, Range:0--4096
______________________________________________________________________________"""
    
    with open('./layers.config', 'w') as f:
        f.write(config_content)
    print("[INFO] 已创建ResNet50测试配置（5层）")


# =========================
# 主函数
# =========================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Pre-profiling examples')
    parser.add_argument('--example', type=int, default=1, 
                       choices=[1, 2, 3, 0],
                       help='选择示例: 1=基础测量, 2=仅探测映射, 3=详细测量, 0=全部运行')
    
    args = parser.parse_args()
    
    # 检查 NPU
    if not torch.npu.is_available():
        print("[ERROR] NPU 不可用，请检查 torch_npu 安装")
        sys.exit(1)
    
    print(f"[INFO] NPU 可用，使用设备: npu:7")
    print(f"[INFO] 当前目录: {os.getcwd()}")
    
    try:
        if args.example == 0:
            example_basic()
            example_probe_only()
            example_detailed()
        elif args.example == 1:
            example_basic()
        elif args.example == 2:
            example_probe_only()
        elif args.example == 3:
            example_detailed()
            
        print("\n" + "=" * 60)
        print("示例运行完成！")
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