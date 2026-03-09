import os
import atexit
import threading
import torch
import torch.nn as nn
import re
from typing import List, Optional, Set, Dict, Callable
from functools import wraps
from dataclasses import dataclass

import torch_npu

from .swapManager import SwapManager
from .module_transfer import ReLU, Cat, Add, AvgPool2d, MaxPool2d


# =========================
# 全局日志控制配置
# =========================
_HOOK_VERBOSE = os.getenv("HOOK_VERBOSE", "0").lower() in ("1", "true", "yes", "on")

_LOG_F = None
_LOG_F_LOCK = threading.Lock() if _HOOK_VERBOSE else None

OP_MAP = (
    nn.Conv2d,
    nn.ReLU, nn.ReLU6, nn.LeakyReLU,
    nn.MaxPool2d,
    nn.AdaptiveAvgPool2d,
    nn.Linear,
    nn.Dropout, nn.Dropout2d,
    nn.BatchNorm2d,
    ReLU, Cat, Add, AvgPool2d, MaxPool2d,
)


def _get_log_f():
    """延迟初始化日志文件"""
    global _LOG_F
    if _LOG_F is None:
        log_path = os.getenv("HOOK_LOG", "train_trace.log")
        _LOG_F = open(log_path, "w", buffering=1)
        atexit.register(lambda: _LOG_F and _LOG_F.close())
    return _LOG_F


def _conditional_log(log_func):
    """条件日志装饰器"""
    if _HOOK_VERBOSE:
        @wraps(log_func)
        def wrapper(*args, **kwargs):
            return log_func(*args, **kwargs)
        return wrapper
    else:
        def noop(*args, **kwargs):
            pass
        noop.__name__ = log_func.__name__
        noop.__doc__ = f"[DISABLED] {log_func.__doc__ or ''}"
        return noop


@_conditional_log
def log_message(message: str) -> None:
    """记录普通消息"""
    with _LOG_F_LOCK:
        print(message, file=_get_log_f())


@_conditional_log
def log_lifecycle(action: str, tensor_id: int, module_name: str = "") -> None:
    """记录生命周期事件"""
    with _LOG_F_LOCK:
        print(
            f"[SWAP-LIFE] {action:10} | Tensor: {tensor_id:4} | Module: {module_name}",
            file=_get_log_f()
        )


class LogController:
    """运行时日志控制器"""
    _enabled: bool = _HOOK_VERBOSE
    _lock = threading.Lock() if _HOOK_VERBOSE else None
    
    @classmethod
    def enable(cls) -> None:
        if not cls._lock:
            cls._lock = threading.Lock()
        with cls._lock:
            cls._enabled = True
    
    @classmethod
    def disable(cls) -> None:
        if cls._lock:
            with cls._lock:
                cls._enabled = False
    
    @classmethod
    def is_enabled(cls) -> bool:
        return cls._enabled


# =========================
# 事件类型定义
# =========================
@dataclass
class TensorEvent:
    """单个tensor迁移事件"""
    tensor_id: int
    event_type: str  # "extract", "d2h", "wait_d2h", "h2d", "wait_h2d"
    tag: str  # "input", "output", "weight" (仅用于extract)
    
    def __repr__(self):
        return f"TensorEvent({self.tensor_id}, {self.event_type}, {self.tag})"


class ModuleEvents:
    """一个模块的所有事件（按执行顺序）"""
    def __init__(self, module_name: str, is_forward: bool):
        self.module_name = module_name
        self.is_forward = is_forward  # True=前向, False=反向
        self.events: List[TensorEvent] = []
    
    def add_event(self, event: TensorEvent):
        self.events.append(event)
    
    def __repr__(self):
        direction = "FWD" if self.is_forward else "BWD"
        return f"ModuleEvents({self.module_name}, {direction}, {len(self.events)} events)"


class HookManager:
    def __init__(self, swap_manager: SwapManager, event_file: str, model: nn.Module, verbose: bool = None):
        """
        Args:
            swap_manager: 交换管理器（必须已绑定到正确 device）
            event_file: 事件文件路径
            model: 目标模型
            verbose: 是否打印日志
        """
        self.swap_manager = swap_manager
        self.model = model
        self.device = swap_manager.device
        
        # 日志控制
        self._verbose_override = verbose
        if verbose is not None:
            if verbose:
                LogController.enable()
            else:
                LogController.disable()
        self._local_verbose = LogController.is_enabled()

        # 加载原始事件
        self.raw_events = self._load_raw_events(event_file)
        
        # 模块事件映射：module_name -> ModuleEvents
        self.forward_events: Dict[str, ModuleEvents] = {}
        self.backward_events: Dict[str, ModuleEvents] = {}
        
        # hook句柄
        self.hooks = []
        self._probe_finished = False
        
        self._conditional_log_init()

    def _conditional_log_init(self):
        if self._local_verbose:
            log_message(f"[HOOK-INIT] Loaded {len(self.raw_events)} raw events")
            log_message(f"[HOOK-INIT] Bound to device: {self.device}")

    def _load_raw_events(self, file_path: str) -> List[Dict]:
        """加载原始事件文件"""
        pattern = re.compile(
            r"Issued Time: (\d+) Tensor: (\d+) From: (\w+), To: (\w+) tag: (\w+)"
        )
        events = []
        with open(file_path, "r") as f:
            for line in f:
                m = pattern.match(line.strip())
                if m:
                    events.append({
                        'issued_time': int(m.group(1)),
                        'tensor_id': int(m.group(2)),
                        'from': m.group(3),
                        'to': m.group(4),
                        'tag': m.group(5),
                    })
        return events

    # =========================
    # 探测阶段：建立模块->事件映射
    # =========================
    def probe_modules(self, sample_input: torch.Tensor, sample_target: Optional[torch.Tensor] = None, criterion: Optional[nn.Module] = None):
        """
        探测阶段：运行完整前向+反向，记录每个issued_time对应的模块，
        然后建立模块->事件列表的映射
        """
        if self._probe_finished:
            return

        # 准备输入
        if sample_target is None:
            sample_target = torch.randint(0, 10, (sample_input.size(0),))
            sample_target = sample_target.to(f"npu:{self.device}")
        else:
            sample_target = sample_target.to(f"npu:{self.device}")
            
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        # 探测用的临时hook
        forward_times: Dict[int, str] = {}  # issued_time -> module_name
        backward_times: Dict[int, str] = {}  # issued_time -> module_name
        
        forward_hooks = []
        backward_hooks = []

        def make_forward_probe(name):
            def hook_fn(m, inp, out):
                # 使用简单计数器代替issued_time
                t = len(forward_times) + 1
                forward_times[t] = name
            return hook_fn

        def make_backward_probe(name):
            def hook_fn(m, grad_in, grad_out):
                t = len(backward_times) + 1
                backward_times[t] = name
            return hook_fn

        # 注册探测hook
        with torch.npu.device(self.device):
            for name, module in self.model.named_modules():
                if isinstance(module, OP_MAP):
                    forward_hooks.append(module.register_forward_hook(make_forward_probe(name)))
                    backward_hooks.append(module.register_full_backward_hook(make_backward_probe(name)))

        try:
            # 运行前向+反向
            output = self.model(sample_input)
            loss = criterion(output, sample_target)
            loss.backward()
            
            # 建立事件映射
            self._build_event_mapping(forward_times, backward_times)
            
            if self._local_verbose:
                total_fwd = sum(len(me.events) for me in self.forward_events.values())
                total_bwd = sum(len(me.events) for me in self.backward_events.values())
                log_message(f"[HOOK-PROBE] Forward: {len(self.forward_events)} modules, {total_fwd} events")
                log_message(f"[HOOK-PROBE] Backward: {len(self.backward_events)} modules, {total_bwd} events")
                
        finally:
            for h in forward_hooks:
                h.remove()
            for h in backward_hooks:
                h.remove()
            
            self._probe_finished = True

    def _build_event_mapping(self, forward_times: Dict[int, str], backward_times: Dict[int, str]):
        """
        根据探测到的时间-模块映射，将原始事件分配到对应模块
        """
        # 处理前向事件
        for raw_ev in self.raw_events:
            t = raw_ev['issued_time']
            if t in forward_times:
                module_name = forward_times[t]
                
                if module_name not in self.forward_events:
                    self.forward_events[module_name] = ModuleEvents(module_name, is_forward=True)
                
                # 确定事件类型
                event_type = self._determine_event_type(raw_ev['from'], raw_ev['to'])
                tensor_ev = TensorEvent(
                    tensor_id=raw_ev['tensor_id'],
                    event_type=event_type,
                    tag=raw_ev['tag']
                )
                self.forward_events[module_name].add_event(tensor_ev)
                
            elif t in backward_times:
                module_name = backward_times[t]
                
                if module_name not in self.backward_events:
                    self.backward_events[module_name] = ModuleEvents(module_name, is_forward=False)
                
                event_type = self._determine_event_type(raw_ev['from'], raw_ev['to'])
                tensor_ev = TensorEvent(
                    tensor_id=raw_ev['tensor_id'],
                    event_type=event_type,
                    tag=raw_ev['tag']
                )
                self.backward_events[module_name].add_event(tensor_ev)

    def _determine_event_type(self, from_loc: str, to_loc: str) -> str:
        """根据from/to位置确定事件类型"""
        if from_loc == "Not_Known" and to_loc == "In_gpu":
            return "extract"
        elif from_loc == "In_gpu" and to_loc == "In_cpu":
            return "d2h"
        elif from_loc == "In_cpu" and to_loc == "In_cpu":
            return "wait_d2h"
        elif from_loc == "In_cpu" and to_loc == "In_gpu":
            return "h2d"
        elif from_loc == "In_gpu" and to_loc == "In_gpu":
            return "wait_h2d"
        else:
            raise ValueError(f"Unknown transition: {from_loc} -> {to_loc}")

    def register_hooks(self):
        """根据探测结果，在需要的模块上注册hook"""
        if not self._probe_finished:
            raise RuntimeError("Must call probe_modules() first")

        self.remove_hooks()
        
        with torch.npu.device(self.device):
            for name, module in self.model.named_modules():
                # 注册前向hook
                if name in self.forward_events:
                    events = self.forward_events[name]
                    hook_fn = self._make_forward_hook(name, events)
                    self.hooks.append(module.register_forward_hook(hook_fn))
                
                # 注册反向hook
                if name in self.backward_events:
                    events = self.backward_events[name]
                    hook_fn = self._make_backward_hook(name, events)
                    self.hooks.append(module.register_full_backward_hook(hook_fn))

        if self._local_verbose:
            log_message(f"[HOOK-REG] Registered {len(self.hooks)} hooks total")

    def setup(self, sample_input: torch.Tensor, sample_target: Optional[torch.Tensor] = None, criterion: Optional[nn.Module] = None):
        """一键完成探测和注册"""
        self.probe_modules(sample_input, sample_target, criterion)
        self.register_hooks()

    # =========================
    # 核心：创建带预绑定事件的hook函数
    # =========================
    def _make_forward_hook(self, module_name: str, module_events: ModuleEvents):
        """
        创建前向hook，事件列表已预绑定，无需issued_time判断
        """
        events = module_events.events
        swap_mgr = self.swap_manager
        device = self.device
        verbose = self._local_verbose
        model = self.model

        def forward_hook(module, inputs, output):
            # 直接在正确的device上下文中获取stream
            with torch.npu.device(device):
                compute_stream = torch_npu.npu.current_stream()
            
            # 按顺序处理所有预绑定的事件
            for ev in events:
                tid = ev.tensor_id
                
                if ev.event_type == "extract":
                    # 提取tensor
                    tensor = None
                    if ev.tag == "input":
                        tensor = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
                    elif ev.tag == "output":
                        tensor = output
                    elif ev.tag == "weight":
                        mod = model.get_submodule(module_name)
                        tensor = self._get_module_weights(mod)
                    
                    if tensor is not None:
                        swap_mgr.add_swap_tensor(tid, tensor)
                        if verbose:
                            log_lifecycle("extract", tid, module_name)
                
                elif ev.event_type == "d2h":
                    swap_mgr.launch_d2h(tid, compute_stream)
                    if verbose:
                        log_lifecycle("d2h", tid, module_name)
                
                elif ev.event_type == "wait_d2h":
                    swap_mgr.wait_d2h_finished(tid, compute_stream)
                    if verbose:
                        log_lifecycle("wait_d2h", tid, module_name)
                
                elif ev.event_type == "h2d":
                    swap_mgr.launch_h2d(tid, compute_stream)
                    if verbose:
                        log_lifecycle("h2d", tid, module_name)
                
                elif ev.event_type == "wait_h2d":
                    swap_mgr.wait_h2d_finished(tid, compute_stream)
                    if verbose:
                        log_lifecycle("wait_h2d", tid, module_name)

        return forward_hook

    def _make_backward_hook(self, module_name: str, module_events: ModuleEvents):
        """
        创建反向hook，事件列表已预绑定
        """
        events = module_events.events
        swap_mgr = self.swap_manager
        device = self.device
        verbose = self._local_verbose

        def backward_hook(module, grad_input, grad_output):
            with torch.npu.device(device):
                compute_stream = torch_npu.npu.current_stream()
            
            # 按顺序处理所有预绑定的事件
            for ev in events:
                tid = ev.tensor_id
                
                if ev.event_type == "d2h":
                    swap_mgr.launch_d2h(tid, compute_stream)
                    if verbose:
                        log_lifecycle("d2h", tid, module_name)
                
                elif ev.event_type == "wait_d2h":
                    swap_mgr.wait_d2h_finished(tid, compute_stream)
                    if verbose:
                        log_lifecycle("wait_d2h", tid, module_name)
                
                elif ev.event_type == "h2d":
                    swap_mgr.launch_h2d(tid, compute_stream)
                    if verbose:
                        log_lifecycle("h2d", tid, module_name)
                
                elif ev.event_type == "wait_h2d":
                    swap_mgr.wait_h2d_finished(tid, compute_stream)
                    if verbose:
                        log_lifecycle("wait_h2d", tid, module_name)
                
                # 注意：反向一般不需要extract，因为tensor已在之前注册

        return backward_hook

    def _get_module_weights(self, module: nn.Module) -> Optional[torch.Tensor]:
        """获取模块的weight和bias"""
        if module is None:
            return None
        weights = []
        for name in ("weight", "bias"):
            p = getattr(module, name, None)
            if p is not None:
                weights.append(p.view(-1))
        return torch.cat(weights) if weights else None

    def remove_hooks(self):
        """移除所有hook"""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        
        if self._local_verbose:
            log_message("[HOOK] All hooks removed")

    def reset(self):
        """重置状态（每个batch开始时调用）"""
        # 新设计不需要issued_time，此方法保留用于兼容性
        pass


# =========================
# 便捷的模块级接口
# =========================
def enable_hook_logging():
    LogController.enable()


def disable_hook_logging():
    LogController.disable()


def is_hook_logging_enabled() -> bool:
    return LogController.is_enabled()