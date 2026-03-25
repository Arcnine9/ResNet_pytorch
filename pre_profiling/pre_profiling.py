"""
pre_profiling.py
专职：测量各module的前向/反向传播时间
"""

import os
import re
import time
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import json
import warnings

# 修复导入：使用 try/except 兼容包模式和直接运行
try:
    # 作为包导入时
    from .layer_config_parser import LayerConfigParser, LayerInfo
except ImportError:
    # 直接运行时（standalone模式）
    from layer_config_parser import LayerConfigParser, LayerInfo


@dataclass  
class ProfilingResult:
    """单个module的profiling结果"""
    hook_id: int
    module_name: str
    layer_type: str
    
    # 前向传播时间 (微秒)
    forward_time_mean: float
    forward_time_std: float
    forward_times: List[float]
    
    # 反向传播时间 (微秒)
    backward_time_mean: float
    backward_time_std: float
    backward_times: List[float]
    
    # 输入/输出shape
    input_shape: Optional[Tuple[int, ...]]
    output_shape: Optional[Tuple[int, ...]]
    
    # 参数数量
    param_count: int = 0
    
    # 额外信息
    device: str = "cpu"
    num_runs: int = 5
    
    def to_line(self) -> str:
        """输出为文本行格式"""
        input_str = str(self.input_shape) if self.input_shape else "None"
        output_str = str(self.output_shape) if self.output_shape else "None"
        return (f"Hook:{self.hook_id} | {self.module_name} | Type:{self.layer_type} | "
                f"FWD:{self.forward_time_mean:.2f}±{self.forward_time_std:.2f}us | "
                f"BWD:{self.backward_time_mean:.2f}±{self.backward_time_std:.2f}us | "
                f"In:{input_str} | Out:{output_str} | Params:{self.param_count}")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
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
    预profiling器：测量各module的前向/反向传播时间
    """
    
    def __init__(self, model: nn.Module, layer_config_path: str, device: Union[str, int] = 'npu:0'):
        self.model = model
        self.device_str = str(device)
        self.device = self._parse_device(device)
        
        # 解析layer config
        self.layer_info_dict = LayerConfigParser.parse(layer_config_path)
        
        # 建立映射
        self.module_to_layer_info: Dict[str, LayerInfo] = {}
        self.results: Dict[str, ProfilingResult] = {}
        
        # 自动检测设备类型
        self.device_type = 'npu' if 'npu' in self.device_str else ('cuda' if 'cuda' in self.device_str else 'cpu')
        
    def _parse_device(self, device: Union[str, int]) -> torch.device:
        """解析设备字符串"""
        if isinstance(device, int):
            return torch.device(f'npu:{device}')
        return torch.device(device)
    
    def _synchronize(self):
        """同步设备"""
        if self.device_type == 'npu':
            if hasattr(torch, 'npu') and torch.npu.is_available():
                torch.npu.synchronize()
        elif self.device_type == 'cuda':
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    
    def _create_dummy_input(self, layer_info: LayerInfo) -> Optional[torch.Tensor]:
        """根据layer_info创建dummy输入"""
        shape = layer_info.get_input_shape()
        
        if not shape:
            # 尝试从第一个input tensor推断
            for t in layer_info.input_tensors:
                if not t.get('is_weight', False):
                    size = t.get('size', 0)
                    if size > 0:
                        batch = layer_info.get_batch_size() or 1
                        shape = (batch, 3, 32, 32)
                        break
        
        if shape:
            try:
                return torch.randn(*shape, device=self.device)
            except Exception as e:
                warnings.warn(f"Failed to create input with shape {shape}: {e}")
                return None
        
        return None
    
    def _match_modules_to_layers(self) -> Dict[int, Tuple[str, nn.Module]]:
        """建立hook_id到module的映射"""
        hook_id_to_module: Dict[int, Tuple[str, nn.Module]] = {}
        
        # 收集所有叶子模块
        leaf_modules = []
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0 or isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                leaf_modules.append((name, module))
        
        # 按hook_id排序并匹配
        sorted_hook_ids = sorted(self.layer_info_dict.keys())
        
        for i, hook_id in enumerate(sorted_hook_ids):
            if i < len(leaf_modules):
                name, module = leaf_modules[i]
                hook_id_to_module[hook_id] = (name, module)
                self.module_to_layer_info[name] = self.layer_info_dict[hook_id]
        
        return hook_id_to_module
    
    def _measure_forward(self, module: nn.Module, dummy_input: torch.Tensor, 
                        num_runs: int = 5) -> Tuple[float, float, List[float]]:
        """测量前向传播时间"""
        times = []
        
        # Warmup
        try:
            _ = module(dummy_input)
            self._synchronize()
        except Exception as e:
            raise RuntimeError(f"Forward warmup failed: {e}")
        
        # 正式测量
        for _ in range(num_runs):
            self._synchronize()
            start = time.perf_counter()
            
            output = module(dummy_input)
            
            self._synchronize()
            end = time.perf_counter()
            
            times.append((end - start) * 1e6)  # 微秒
        
        mean_time = sum(times) / len(times)
        std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5 if len(times) > 1 else 0
        
        return mean_time, std_time, times
    
    def _measure_backward(self, module: nn.Module, dummy_input: torch.Tensor,
                         num_runs: int = 5) -> Tuple[float, float, List[float]]:
        """测量反向传播时间"""
        times = []
        
        # Warmup
        try:
            input_var = dummy_input.clone().requires_grad_(True)
            output = module(input_var)
            grad_output = torch.randn_like(output)
            output.backward(grad_output)
            self._synchronize()
        except Exception as e:
            return 0, 0, [0] * num_runs
        
        # 正式测量
        for _ in range(num_runs):
            input_var = dummy_input.clone().requires_grad_(True)
            
            self._synchronize()
            start = time.perf_counter()
            
            output = module(input_var)
            grad_output = torch.randn_like(output)
            output.backward(grad_output)
            
            self._synchronize()
            end = time.perf_counter()
            
            times.append((end - start) * 1e6)
        
        mean_time = sum(times) / len(times)
        std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5 if len(times) > 1 else 0
        
        return mean_time, std_time, times
    
    def run(self, num_runs: int = 5, warmup: int = 2, 
            measure_backward: bool = True) -> Dict[str, ProfilingResult]:
        """运行profiling"""
        print(f"[PreProfiler] Starting profiling on {self.device}")
        print(f"[PreProfiler] Configuration: {warmup} warmup + {num_runs} runs")
        
        hook_id_to_module = self._match_modules_to_layers()
        print(f"[PreProfiler] Matched {len(hook_id_to_module)} modules with layer config")
        
        # 将模型移到设备
        self.model.to(self.device)
        self.model.train()  # BN等层需要train模式
        
        # 对每个module进行profiling
        for hook_id in sorted(hook_id_to_module.keys()):
            module_name, module = hook_id_to_module[hook_id]
            layer_info = self.layer_info_dict[hook_id]
            
            dummy_input = self._create_dummy_input(layer_info)
            if dummy_input is None:
                print(f"[PreProfiler-WARN] Cannot create input for {module_name} (hook:{hook_id}), skipping")
                continue
            
            print(f"[PreProfiler] Profiling {module_name} (hook:{hook_id})...", end=' ')
            
            try:
                # Warmup
                for _ in range(warmup):
                    _ = module(dummy_input)
                
                # 测量前向
                fwd_mean, fwd_std, fwd_times = self._measure_forward(module, dummy_input, num_runs)
                
                # 测量反向
                if measure_backward:
                    bwd_mean, bwd_std, bwd_times = self._measure_backward(module, dummy_input, num_runs)
                else:
                    bwd_mean = bwd_std = 0
                    bwd_times = [0] * num_runs
                
                # 计算参数数量
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
        
        print(f"[PreProfiler] Profiling complete: {len(self.results)}/{len(hook_id_to_module)} modules successful")
        return self.results
    
    def save_results(self, output_path: str = 'module_profiling.txt', format: str = 'text'):
        """保存profiling结果"""
        if format == 'json':
            self._save_json(output_path)
        else:
            self._save_text(output_path)
    
    def _save_text(self, output_path: str):
        """保存为文本格式"""
        with open(output_path, 'w') as f:
            f.write("# Module Profiling Results\n")
            f.write(f"# Generated on device: {self.device}\n")
            f.write(f"# Format: Hook:ID | ModuleName | Type:LayerType | FWD:Mean±Std(us) | BWD:Mean±Std(us) | In:InputShape | Out:OutputShape | Params:Count\n")
            f.write("# Raw times (us) for each run are listed after each line\n")
            f.write("=" * 100 + "\n\n")
            
            for module_name in sorted(self.results.keys()):
                result = self.results[module_name]
                f.write(result.to_line() + "\n")
                f.write(f"# RAW_FWD: {result.forward_times}\n")
                f.write(f"# RAW_BWD: {result.backward_times}\n")
                f.write("\n")
        
        print(f"[PreProfiler] Results saved to {output_path}")
    
    def _save_json(self, output_path: str):
        """保存为JSON格式"""
        data = {
            'device': str(self.device),
            'num_modules': len(self.results),
            'modules': {name: result.to_dict() for name, result in self.results.items()}
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[PreProfiler] Results saved to {output_path}")
    
    def get_results_summary(self) -> Dict[str, Any]:
        """获取结果摘要"""
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
            'fastest_module': min(self.results.items(), key=lambda x: x[1].forward_time_mean + x[1].backward_time_mean)
        }


def run_pre_profiling(model: nn.Module, 
                     layer_config_path: str,
                     output_path: str = 'module_profiling.txt',
                     device: Union[str, int] = 'npu:0',
                     num_runs: int = 5,
                     warmup: int = 2,
                     measure_backward: bool = True,
                     output_format: str = 'text') -> Dict[str, ProfilingResult]:
    """
    便捷的预profiling接口
    """
    profiler = PreProfiler(model, layer_config_path, device)
    results = profiler.run(num_runs=num_runs, warmup=warmup, measure_backward=measure_backward)
    profiler.save_results(output_path, format=output_format)
    
    # 打印摘要
    summary = profiler.get_results_summary()
    if summary:
        print(f"\n[Summary] Total modules: {summary['num_modules']}")
        print(f"[Summary] Total forward time: {summary['total_forward_time_us']/1000:.2f} ms")
        print(f"[Summary] Total backward time: {summary['total_backward_time_us']/1000:.2f} ms")
        slow_name, slow_result = summary['slowest_module']
        print(f"[Summary] Slowest module: {slow_name} ({slow_result.forward_time_mean + slow_result.backward_time_mean:.2f} us)")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Pre-profiling for neural network modules')
    parser.add_argument('config', type=str, help='Path to layer.config file')
    parser.add_argument('--model', type=str, default='inception_v3', 
                       choices=['inception_v3', 'resnet50', 'custom'],
                       help='Model name')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu, cuda:0, npu:0, etc.)')
    parser.add_argument('--runs', type=int, default=5, help='Number of measurement runs')
    parser.add_argument('--warmup', type=int, default=2, help='Number of warmup runs')
    parser.add_argument('--output', type=str, default='module_profiling.txt', help='Output file path')
    parser.add_argument('--format', type=str, default='text', choices=['text', 'json'],
                       help='Output format')
    parser.add_argument('--no-backward', action='store_true', help='Skip backward measurement')
    
    args = parser.parse_args()
    
    # 加载模型
    if args.model == 'inception_v3':
        from torchvision.models import inception_v3
        model = inception_v3(aux_logits=False, init_weights=True)
    elif args.model == 'resnet50':
        from torchvision.models import resnet50
        model = resnet50(weights=None)
    else:
        print("Custom model not supported in CLI mode")
        exit(1)
    
    # 运行profiling
    results = run_pre_profiling(
        model=model,
        layer_config_path=args.config,
        output_path=args.output,
        device=args.device,
        num_runs=args.runs,
        warmup=args.warmup,
        measure_backward=not args.no_backward,
        output_format=args.format
    )
    
    print(f"\nProfiling complete. Results saved to {args.output}")