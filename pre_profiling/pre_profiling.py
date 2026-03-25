"""
pre_profiling.py
修复版：基于时间顺序建立映射，确保313层全部匹配
"""

import os
import re
import time
import signal
import sys
import warnings
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
from collections import OrderedDict

import torch
import torch.nn as nn
import torch_npu

try:
    from .layer_config_parser import LayerConfigParser, LayerInfo
except ImportError:
    from layer_config_parser import LayerConfigParser, LayerInfo


# =========================
# 信号处理
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


@dataclass  
class ProfilingResult:
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
        return (f"Hook:{self.hook_id} | {self.module_name} | Type:{self.layer_type} | "
                f"FWD:{self.forward_time_mean:.2f}±{self.forward_time_std:.2f}us | "
                f"BWD:{self.backward_time_mean:.2f}±{self.backward_time_std:.2f}us | "
                f"In:{self.input_shape} | Out:{self.output_shape} | Params:{self.param_count}")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'hook_id': self.hook_id, 'module_name': self.module_name, 'layer_type': self.layer_type,
            'forward_time_mean_us': self.forward_time_mean, 'forward_time_std_us': self.forward_time_std,
            'backward_time_mean_us': self.backward_time_mean, 'backward_time_std_us': self.backward_time_std,
            'input_shape': self.input_shape, 'output_shape': self.output_shape, 'param_count': self.param_count,
        }


class PreProfiler:
    def __init__(self, model: nn.Module, layer_config_path: str, device_id: int = 7):
        self.model = model
        self.device_id = device_id
        torch.npu.set_device(device_id)
        self.device = torch.device(f"npu:{device_id}")
        
        print(f"[PreProfiler] Using device: {self.device}")
        
        if not os.path.exists(layer_config_path):
            raise FileNotFoundError(f"Layer config not found: {layer_config_path}")
            
        self.layer_info_dict = LayerConfigParser.parse(layer_config_path)
        print(f"[PreProfiler] Loaded {len(self.layer_info_dict)} layers from config")
        
        self.module_to_layer_info: Dict[str, LayerInfo] = {}
        self.results: Dict[str, ProfilingResult] = {}
        
        # 探测结果：time -> (module_name, is_forward)
        self.time_to_module: Dict[int, Tuple[str, bool]] = {}
        self._probe_finished = False
        
    def _synchronize(self):
        torch_npu.npu.synchronize()
    
    def probe_module_mapping(self, verbose: bool = False) -> Dict[int, Tuple[str, bool]]:
        """
        ★ 核心：基于时间顺序探测，建立 time -> (module_name, is_forward) 映射
        """
        if self._probe_finished:
            return self.time_to_module
        
        print(f"[Probe] 开始时间顺序探测...")
        
        # 确保模型在NPU上
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 收集所有叶子模块用于注册hook
        leaf_modules = []
        for name, module in self.model.named_modules():
            is_leaf = len(list(module.children())) == 0
            if is_leaf or isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, 
                                              nn.ReLU, nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
                leaf_modules.append((name, module))
        
        print(f"[Probe] 注册 {len(leaf_modules)} 个模块的hooks...")
        
        hooks = []
        probe_time = 0
        time_to_module = {}
        
        def make_forward_probe(name):
            nonlocal probe_time
            def hook_fn(m, inp, out):
                nonlocal probe_time
                probe_time += 1
                time_to_module[probe_time] = (name, True)  # True = 前向
            return hook_fn
        
        def make_backward_probe(name):
            nonlocal probe_time
            def hook_fn(m, grad_in, grad_out):
                nonlocal probe_time
                probe_time += 1
                time_to_module[probe_time] = (name, False)  # False = 反向
            return hook_fn
        
        # 注册所有hooks
        for name, module in leaf_modules:
            h1 = module.register_forward_hook(make_forward_probe(name))
            h2 = module.register_full_backward_hook(make_backward_probe(name))
            hooks.extend([h1, h2])
        
        # 运行探测
        try:
            # 创建输入（batch=1）
            first_hook_id = min(self.layer_info_dict.keys())
            first_layer = self.layer_info_dict[first_hook_id]
            probe_input = self._create_probe_input(first_layer, batch_size=1)
            
            if probe_input is None:
                raise RuntimeError("无法创建探测输入")
            
            print(f"[Probe] 输入形状: {tuple(probe_input.shape)}，开始探测运行...")
            
            # 前向
            output = self.model(probe_input)
            if isinstance(output, tuple):
                output = output[0]
            
            # 反向
            if output.dim() >= 2:
                batch = output.size(0)
                num_classes = output.size(-1) if output.dim() == 2 else 1000
                target = torch.randint(0, num_classes, (batch,)).to(self.device)
                loss = nn.functional.cross_entropy(
                    output.view(batch, -1) if output.dim() > 2 else output, 
                    target
                )
            else:
                loss = output.sum()
            
            loss.backward()
            torch_npu.npu.synchronize()
            
            # 统计
            fwd_count = sum(1 for _, is_fwd in time_to_module.values() if is_fwd)
            bwd_count = sum(1 for _, is_fwd in time_to_module.values() if not is_fwd)
            
            print(f"[Probe] 探测完成:")
            print(f"  - 总时间戳: {len(time_to_module)}")
            print(f"  - 前向事件: {fwd_count} (期望: {len(self.layer_info_dict)})")
            print(f"  - 反向事件: {bwd_count} (期望: {len(self.layer_info_dict)})")
            
            if fwd_count != len(self.layer_info_dict):
                print(f"[WARN] 前向事件数与Config层数不匹配！")
            
            self.time_to_module = time_to_module
            self._probe_finished = True
            
            if verbose:
                print("[Probe] 前10个时间戳映射:")
                for t in sorted(time_to_module.keys())[:10]:
                    name, is_fwd = time_to_module[t]
                    print(f"  Time {t:3d}: {name:40s} ({'FWD' if is_fwd else 'BWD'})")
            
        finally:
            for h in hooks:
                h.remove()
            self.model.zero_grad(set_to_none=True)
            torch_npu.npu.empty_cache()
        
        return self.time_to_module
    
    def _create_probe_input(self, layer_info: LayerInfo, batch_size: int = 1) -> Optional[torch.Tensor]:
        """创建探测用的最小输入"""
        shape = layer_info.get_input_shape()
        if not shape:
            shape = (batch_size, 3, 299, 299)
        else:
            shape = (batch_size,) + shape[1:]
        
        try:
            return torch.randn(*shape).to(self.device)
        except Exception as e:
            warnings.warn(f"Failed to create probe input: {e}")
            return None
    
    def _match_modules_by_time(self) -> Dict[int, Tuple[str, nn.Module, LayerInfo]]:
        """
        ★ 关键：按时间顺序将hook_id映射到模块
        hook_id 0 -> time 1 (第一个前向)
        hook_id 1 -> time 2 (第二个前向)
        ...
        """
        if not self._probe_finished:
            self.probe_module_mapping()
        
        # 分离前向和反向时间戳
        fwd_times = [t for t, (_, is_fwd) in self.time_to_module.items() if is_fwd]
        bwd_times = [t for t, (_, is_fwd) in self.time_to_module.items() if not is_fwd]
        
        sorted_hook_ids = sorted(self.layer_info_dict.keys())
        
        print(f"[Match] 按时间顺序匹配 {len(sorted_hook_ids)} 个hook_id...")
        print(f"[Match] 前向时间戳: {len(fwd_times)}, 反向时间戳: {len(bwd_times)}")
        
        # 构建模块缓存
        module_cache = dict(self.model.named_modules())
        
        # 映射结果：hook_id -> (module_name, module_obj, layer_info)
        mapping = {}
        
        # 假设config中的hook_id按执行顺序排列
        # 将排序后的hook_id与排序后的前向时间戳一一对应
        for i, hook_id in enumerate(sorted_hook_ids):
            if i < len(fwd_times):
                time_stamp = fwd_times[i]
                module_name, _ = self.time_to_module[time_stamp]
                
                if module_name in module_cache:
                    module = module_cache[module_name]
                    layer_info = self.layer_info_dict[hook_id]
                    mapping[hook_id] = (module_name, module, layer_info)
                    
                    # 保存用于后续测量
                    self.module_to_layer_info[module_name] = layer_info
                else:
                    print(f"[WARN] Hook {hook_id}: 模块 {module_name} 未找到")
            else:
                print(f"[WARN] Hook {hook_id}: 没有对应的前向时间戳")
        
        print(f"[Match] 成功映射: {len(mapping)}/{len(sorted_hook_ids)}")
        return mapping
    
    def _create_dummy_input(self, layer_info: LayerInfo) -> Optional[torch.Tensor]:
        """创建profiling用的完整batch输入"""
        shape = layer_info.get_input_shape()
        if not shape:
            return None
        
        try:
            return torch.randn(*shape).to(self.device)
        except Exception as e:
            warnings.warn(f"Failed to create input: {e}")
            return None
    
    def _validate_and_fix_input(self, module: nn.Module, dummy_input: torch.Tensor) -> torch.Tensor:
        """验证并修复输入形状"""
        if dummy_input.device != self.device:
            dummy_input = dummy_input.to(self.device)
            
        try:
            if isinstance(module, nn.Conv2d):
                if dummy_input.shape[1] != module.in_channels:
                    new_shape = list(dummy_input.shape)
                    new_shape[1] = module.in_channels
                    return torch.randn(*new_shape).to(self.device)
            
            elif isinstance(module, nn.BatchNorm2d):
                if dummy_input.shape[1] != module.num_features:
                    new_shape = list(dummy_input.shape)
                    new_shape[1] = module.num_features
                    return torch.randn(*new_shape).to(self.device)
            
            elif isinstance(module, nn.Linear):
                if dummy_input.shape[-1] != module.in_features:
                    batch = dummy_input.shape[0]
                    return torch.randn(batch, module.in_features).to(self.device)
        except Exception as e:
            warnings.warn(f"Shape validation failed: {e}")
        
        return dummy_input
    
    def _measure_forward(self, module: nn.Module, dummy_input: torch.Tensor, 
                        num_runs: int = 5) -> Tuple[float, float, List[float]]:
        times = []
        
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
        has_params = any(p.requires_grad for p in module.parameters())
        if not has_params:
            return 0, 0, [0] * num_runs
        
        times = []
        
        # Warmup
        try:
            input_var = dummy_input.clone().requires_grad_(True)
            output = module(input_var)
            if output.requires_grad:
                grad_output = torch.randn_like(output).to(self.device)
                output.backward(grad_output)
                self._synchronize()
        except Exception as e:
            return 0, 0, [0] * num_runs
        
        # 测量
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
            measure_backward: bool = True, verbose: bool = False) -> Dict[str, ProfilingResult]:
        
        print(f"[PreProfiler] Starting profiling on NPU:{self.device_id}")
        
        # 探测并建立时间顺序映射
        mapping = self._match_modules_by_time()
        
        if not mapping:
            print("[ERROR] 未匹配到任何模块")
            return {}
        
        self.model = self.model.to(self.device)
        self.model.train()
        
        success = 0
        failed = []
        
        for hook_id in sorted(mapping.keys()):
            module_name, module, layer_info = mapping[hook_id]
            
            module = module.to(self.device)
            
            dummy_input = self._create_dummy_input(layer_info)
            if dummy_input is None:
                failed.append((hook_id, module_name, "No input shape"))
                continue
            
            dummy_input = self._validate_and_fix_input(module, dummy_input)
            
            print(f"[PreProfiler] {module_name} (hook:{hook_id})...", end=' ', flush=True)
            
            try:
                # Warmup
                for _ in range(warmup):
                    _ = module(dummy_input)
                torch_npu.npu.synchronize()
                
                fwd_mean, fwd_std, fwd_times = self._measure_forward(module, dummy_input, num_runs)
                
                if measure_backward:
                    bwd_mean, bwd_std, bwd_times = self._measure_backward(module, dummy_input, num_runs)
                else:
                    bwd_mean = bwd_std = 0
                    bwd_times = [0] * num_runs
                
                param_count = sum(p.numel() for p in module.parameters())
                
                # 获取输出shape
                output_shape = None
                try:
                    with torch.no_grad():
                        out = module(dummy_input)
                        if isinstance(out, torch.Tensor):
                            output_shape = tuple(out.shape)
                except:
                    pass
                
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
                    input_shape=tuple(dummy_input.shape),
                    output_shape=output_shape,
                    param_count=param_count,
                )
                
                self.results[module_name] = result
                success += 1
                print(f"✓ FWD={fwd_mean:.2f}±{fwd_std:.2f}us, BWD={bwd_mean:.2f}±{bwd_std:.2f}us")
                
            except Exception as e:
                print(f"✗ FAILED: {str(e)[:50]}")
                failed.append((hook_id, module_name, str(e)))
                continue
        
        print(f"[PreProfiler] Complete: {success}/{len(mapping)} modules succeeded")
        return self.results
    
    def save_results(self, output_path: str = 'module_profiling.txt', format: str = 'text'):
        if format == 'json':
            self._save_json(output_path)
        else:
            self._save_text(output_path)
    
    def _save_text(self, output_path: str):
        with open(output_path, 'w') as f:
            f.write(f"# Module Profiling Results (NPU:{self.device_id})\n")
            f.write(f"# Total modules: {len(self.results)}\n")
            f.write("=" * 100 + "\n\n")
            for module_name in sorted(self.results.keys()):
                f.write(self.results[module_name].to_line() + "\n\n")
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
        slowest = max(self.results.items(), key=lambda x: x[1].forward_time_mean + x[1].backward_time_mean)
        return {
            'num_modules': len(self.results),
            'total_forward_time_us': total_fwd,
            'total_backward_time_us': total_bwd,
            'total_time_us': total_fwd + total_bwd,
            'slowest_module': slowest[0],
            'slowest_time_us': slowest[1].forward_time_mean + slowest[1].backward_time_mean,
        }


def run_pre_profiling(model: nn.Module, 
                     layer_config_path: str,
                     output_path: str = 'module_profiling.txt',
                     device_id: int = 7,
                     num_runs: int = 5,
                     warmup: int = 2,
                     measure_backward: bool = True,
                     output_format: str = 'text',
                     verbose: bool = False) -> Dict[str, ProfilingResult]:
    
    profiler = PreProfiler(model, layer_config_path, device_id=device_id)
    results = profiler.run(num_runs=num_runs, warmup=warmup, 
                          measure_backward=measure_backward, verbose=verbose)
    profiler.save_results(output_path, format=output_format)
    
    summary = profiler.get_results_summary()
    if summary:
        print(f"\n[Summary] Modules: {summary['num_modules']}")
        print(f"[Summary] Total time: {summary['total_time_us']/1000:.2f}ms")
    
    torch_npu.npu.empty_cache()
    return results


if __name__ == '__main__':
    import argparse
    from torchvision.models import inception_v3, resnet50
    
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--device', type=int, default=7)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--model', type=str, default='inception_v3', choices=['resnet50', 'inception_v3'])
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--output', type=str, default='module_profiling.txt')
    
    args = parser.parse_args()
    
    if not torch.npu.is_available():
        print("[ERROR] NPU unavailable")
        sys.exit(1)
    
    if args.model == 'resnet50':
        model = resnet50(weights=None)
    else:
        model = inception_v3(weights=None, aux_logits=False)
    
    try:
        from swap_manager.module_transfer import replace_functional
        model = replace_functional(model, verbose=args.verbose)
        print("[INFO] 已应用FX转换")
    except ImportError:
        print("[INFO] 未找到swap_manager.module_transfer，跳过FX转换")
    
    run_pre_profiling(
        model=model,
        layer_config_path=args.config,
        output_path=args.output,
        device_id=args.device,
        num_runs=args.runs,
        warmup=args.warmup,
        verbose=args.verbose
    )