"""
pre_profiling.py
适配 train.py 风格：默认 npu:7，使用 torch_npu 自动转换
"""

import os
import re
import time
import signal
import sys

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import json
import warnings

# 导入 torch_npu（必须在 torch 之后）
import torch_npu

# 修复导入
try:
    from .layer_config_parser import LayerConfigParser, LayerInfo
except ImportError:
    from layer_config_parser import LayerConfigParser, LayerInfo


# =========================
# 信号处理（同 train.py）
# =========================
def signal_handler(signum, frame):
    print(f"\n[Signal] 捕获信号 {signum}，正在清理并退出...")
    try:
        torch_npu.npu.synchronize()
        torch_npu.npu.empty_cache()
    except:
        pass
    sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


@dataclass  
class ProfilingResult:
    """单个 module 的 profiling 结果"""
    hook_id: int
    module_name: str
    layer_type: str
    
    forward_time_mean: float
    forward_time_std: float
    forward_times: List[float]
    
    backward_time_mean: float
    backward_time_std: float
    backward_times: List[float]
    
    input_shape: Optional[Tuple[int, ...]]
    output_shape: Optional[Tuple[int, ...]]
    param_count: int = 0
    device: str = "npu:7"
    num_runs: int = 5
    
    def to_line(self) -> str:
        input_str = str(self.input_shape) if self.input_shape else "None"
        output_str = str(self.output_shape) if self.output_shape else "None"
        return (f"Hook:{self.hook_id} | {self.module_name} | Type:{self.layer_type} | "
                f"FWD:{self.forward_time_mean:.2f}±{self.forward_time_std:.2f}us | "
                f"BWD:{self.backward_time_mean:.2f}±{self.backward_time_std:.2f}us | "
                f"In:{input_str} | Out:{output_str} | Params:{self.param_count}")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'hook_id': self.hook_id,
            'module_name': self.module_name,
            'layer_type': self.layer_type,
            'forward_time_mean_us': self.forward_time_mean,
            'forward_time_std_us': self.forward_time_std,
            'forward_times_us': self.forward_times,
            'backward_time_mean_us': self.backward_time_mean,
            'backward_time_std_us': self.backward_time_std,
            'backward_times_us': self.backward_times,
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
            'param_count': self.param_count,
            'device': self.device,
            'num_runs': self.num_runs
        }


class PreProfiler:
    """
    预 profiling 器（train.py 风格）
    默认使用 npu:7，与 train.py 保持一致
    """
    
    def __init__(self, model: nn.Module, layer_config_path: str, device_id: int = 7):
        """
        Args:
            device_id: 默认 7（与 train.py 一致）
        """
        self.model = model
        self.device_id = device_id
        
        # 设置设备（同 train.py）
        torch.npu.set_device(device_id)
        self.device = torch.device(f"npu:{device_id}")
        
        print(f"[PreProfiler] Using device: {self.device}")
        
        # 解析 layer config
        self.layer_info_dict = LayerConfigParser.parse(layer_config_path)
        
        self.module_to_layer_info: Dict[str, LayerInfo] = {}
        self.results: Dict[str, ProfilingResult] = {}
        
    def _synchronize(self):
        """同步 NPU"""
        torch_npu.npu.synchronize()
    
    def _create_dummy_input(self, layer_info: LayerInfo) -> Optional[torch.Tensor]:
        """创建 dummy 输入并自动转换到 NPU"""
        shape = layer_info.get_input_shape()
        
        if not shape:
            for t in layer_info.input_tensors:
                if not t.get('is_weight', False):
                    size = t.get('size', 0)
                    if size > 0:
                        batch = layer_info.get_batch_size() or 1
                        shape = (batch, 3, 32, 32)
                        break
        
        if shape:
            try:
                # 创建张量（会自动转换到当前设置的 NPU 设备）
                x = torch.randn(*shape)
                return x.to(self.device)  # 同 train.py 风格
            except Exception as e:
                warnings.warn(f"Failed to create input: {e}")
                return None
        
        return None
    
    def _match_modules_to_layers(self) -> Dict[int, Tuple[str, nn.Module]]:
        """建立 hook_id 到 module 的映射"""
        hook_id_to_module: Dict[int, Tuple[str, nn.Module]] = {}
        
        leaf_modules = []
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0 or isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                leaf_modules.append((name, module))
        
        sorted_hook_ids = sorted(self.layer_info_dict.keys())
        
        for i, hook_id in enumerate(sorted_hook_ids):
            if i < len(leaf_modules):
                name, module = leaf_modules[i]
                hook_id_to_module[hook_id] = (name, module)
                self.module_to_layer_info[name] = self.layer_info_dict[hook_id]
        
        return hook_id_to_module
    
    def _measure_forward(self, module: nn.Module, dummy_input: torch.Tensor, 
                        num_runs: int = 5) -> Tuple[float, float, List[float]]:
        """测量前向（在 torch.npu.device 上下文中）"""
        times = []
        
        # Warmup
        try:
            _ = module(dummy_input)
            self._synchronize()
        except Exception as e:
            raise RuntimeError(f"Forward warmup failed: {e}")
        
        for _ in range(num_runs):
            self._synchronize()
            start = time.perf_counter()
            
            output = module(dummy_input)
            
            self._synchronize()
            end = time.perf_counter()
            
            times.append((end - start) * 1e6)
        
        mean_time = sum(times) / len(times)
        std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5 if len(times) > 1 else 0
        
        return mean_time, std_time, times
    
    def _measure_backward(self, module: nn.Module, dummy_input: torch.Tensor,
                         num_runs: int = 5) -> Tuple[float, float, List[float]]:
        """测量反向（在 torch.npu.device 上下文中）"""
        times = []
        
        # Warmup
        try:
            input_var = dummy_input.clone().requires_grad_(True)
            output = module(input_var)
            grad_output = torch.randn_like(output).to(self.device)
            output.backward(grad_output)
            self._synchronize()
        except Exception as e:
            return 0, 0, [0] * num_runs
        
        for _ in range(num_runs):
            input_var = dummy_input.clone().requires_grad_(True)
            
            self._synchronize()
            start = time.perf_counter()
            
            output = module(input_var)
            grad_output = torch.randn_like(output).to(self.device)
            output.backward(grad_output)
            
            self._synchronize()
            end = time.perf_counter()
            
            times.append((end - start) * 1e6)
        
        mean_time = sum(times) / len(times)
        std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5 if len(times) > 1 else 0
        
        return mean_time, std_time, times
    
    def run(self, num_runs: int = 5, warmup: int = 2, 
            measure_backward: bool = True) -> Dict[str, ProfilingResult]:
        """
        运行 profiling（使用 torch.npu.device 上下文，同 train.py）
        """
        print(f"[PreProfiler] Starting profiling on NPU:{self.device_id}")
        print(f"[PreProfiler] Configuration: {warmup} warmup + {num_runs} runs")
        
        hook_id_to_module = self._match_modules_to_layers()
        print(f"[PreProfiler] Matched {len(hook_id_to_module)} modules")
        
        # 模型移到 NPU（同 train.py: net = net.to(device)）
        self.model = self.model.to(self.device)
        self.model.train()
        
        # 使用 torch.npu.device 上下文（同 train.py）
        with torch.npu.device(self.device_id):
            
            for hook_id in sorted(hook_id_to_module.keys()):
                module_name, module = hook_id_to_module[hook_id]
                layer_info = self.layer_info_dict[hook_id]
                
                # 确保模块在 NPU 上
                module = module.to(self.device)
                
                # 创建输入
                dummy_input = self._create_dummy_input(layer_info)
                if dummy_input is None:
                    print(f"[WARN] Skip {module_name}: no input shape")
                    continue
                
                print(f"[PreProfiler] {module_name} (hook:{hook_id})...", end=' ', flush=True)
                
                try:
                    # Warmup
                    with torch.npu.device(self.device_id):
                        for _ in range(warmup):
                            _ = module(dummy_input)
                        torch_npu.npu.synchronize()
                    
                    # 测量
                    fwd_mean, fwd_std, fwd_times = self._measure_forward(module, dummy_input, num_runs)
                    
                    if measure_backward:
                        bwd_mean, bwd_std, bwd_times = self._measure_backward(module, dummy_input, num_runs)
                    else:
                        bwd_mean = bwd_std = 0
                        bwd_times = [0] * num_runs
                    
                    param_count = sum(p.numel() for p in module.parameters())
                    
                    result = ProfilingResult(
                        hook_id=hook_id,
                        module_name=module_name,
                        layer_type=layer_info.layer_type,
                        forward_time_mean=fwd_mean,
                        forward_time_std=fwd_std,
                        forward_times=fwd_times,
                        backward_time_mean=bwd_mean,
                        backward_time_std=bwd_std,
                        backward_times=bwd_times,
                        input_shape=layer_info.get_input_shape(),
                        output_shape=layer_info.infer_output_shape(layer_info.get_input_shape()),
                        param_count=param_count,
                        device=str(self.device),
                        num_runs=num_runs
                    )
                    
                    self.results[module_name] = result
                    print(f"FWD={fwd_mean:.2f}±{fwd_std:.2f}us, BWD={bwd_mean:.2f}±{bwd_std:.2f}us")
                    
                except Exception as e:
                    print(f"FAILED: {e}")
                    continue
        
        print(f"[PreProfiler] Complete: {len(self.results)}/{len(hook_id_to_module)} modules")
        return self.results
    
    def save_results(self, output_path: str = 'module_profiling.txt', format: str = 'text'):
        """保存结果"""
        if format == 'json':
            self._save_json(output_path)
        else:
            self._save_text(output_path)
    
    def _save_text(self, output_path: str):
        with open(output_path, 'w') as f:
            f.write(f"# Module Profiling Results (NPU:{self.device_id})\n")
            f.write(f"# Device: {self.device}\n")
            f.write("=" * 100 + "\n\n")
            
            for module_name in sorted(self.results.keys()):
                result = self.results[module_name]
                f.write(result.to_line() + "\n")
                f.write(f"# RAW_FWD: {result.forward_times}\n")
                f.write(f"# RAW_BWD: {result.backward_times}\n")
                f.write("\n")
        
        print(f"[PreProfiler] Results saved to {output_path}")
    
    def _save_json(self, output_path: str):
        data = {
            'device': str(self.device),
            'num_modules': len(self.results),
            'modules': {name: r.to_dict() for name, r in self.results.items()}
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[PreProfiler] Results saved to {output_path}")
    
    def get_results_summary(self) -> Dict[str, Any]:
        if not self.results:
            return {}
        
        total_fwd = sum(r.forward_time_mean for r in self.results.values())
        total_bwd = sum(r.backward_time_mean for r in self.results.values())
        
        return {
            'num_modules': len(self.results),
            'total_forward_time_us': total_fwd,
            'total_backward_time_us': total_bwd,
            'total_time_us': total_fwd + total_bwd,
            'slowest_module': max(self.results.items(), key=lambda x: x[1].forward_time_mean + x[1].backward_time_mean),
        }


def run_pre_profiling(model: nn.Module, 
                     layer_config_path: str,
                     output_path: str = 'module_profiling.txt',
                     device_id: int = 7,  # 默认 7，同 train.py
                     num_runs: int = 5,
                     warmup: int = 2,
                     measure_backward: bool = True,
                     output_format: str = 'text') -> Dict[str, ProfilingResult]:
    """
    便捷接口（默认 npu:7）
    """
    profiler = PreProfiler(model, layer_config_path, device_id=device_id)
    results = profiler.run(num_runs=num_runs, warmup=warmup, measure_backward=measure_backward)
    profiler.save_results(output_path, format=output_format)
    
    summary = profiler.get_results_summary()
    if summary:
        print(f"\n[Summary] Modules: {summary['num_modules']}")
        print(f"[Summary] FWD: {summary['total_forward_time_us']/1000:.2f}ms, BWD: {summary['total_backward_time_us']/1000:.2f}ms")
    
    # 清理缓存（同 train.py）
    torch_npu.npu.empty_cache()
    
    return results


if __name__ == '__main__':
    import argparse
    from torchvision.models import inception_v3, resnet50
    
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='layer.config 路径')
    parser.add_argument('--device', type=int, default=7, help='NPU 设备 ID（默认 7）')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--output', type=str, default='module_profiling.txt')
    
    args = parser.parse_args()
    
    # 默认 resnet50（同 train.py 配置）
    model = resnet50(weights=None)
    
    run_pre_profiling(
        model=model,
        layer_config_path=args.config,
        output_path=args.output,
        device_id=args.device,  # 默认 7
        num_runs=args.runs,
        warmup=args.warmup
    )