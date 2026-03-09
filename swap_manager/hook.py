import os
import atexit
import threading
import torch
import torch.nn as nn
import re
from typing import List, Optional, Set, Dict, Tuple
from functools import wraps

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
def log_lifecycle(action: str, tensor_id: int, time: int = 0) -> None:
    """记录生命周期事件"""
    with _LOG_F_LOCK:
        print(
            f"[SWAP-LIFE] {action:10} | Tensor: {tensor_id:4} | Issued: {time:4}",
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
    
    @classmethod
    def toggle(cls) -> bool:
        if not cls._lock:
            cls._lock = threading.Lock()
        with cls._lock:
            cls._enabled = not cls._enabled
            return cls._enabled


def _fast_log_check() -> bool:
    """快速日志检查"""
    return LogController._enabled


class TraceEvent:
    __slots__ = ('issued_time', 'tensor_id', 'from_location', 'to_location', 'tag')
    
    def __init__(self, issued_time: int, tensor_id: int, from_location: str, to_location: str, tag: str):
        self.issued_time = issued_time
        self.tensor_id = tensor_id
        self.from_location = from_location
        self.to_location = to_location
        self.tag = tag


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
        
        # 从 swap_manager 获取 device，确保一致性
        self.device = swap_manager.device
        
        self._verbose_override = verbose
        if verbose is not None:
            if verbose:
                LogController.enable()
            else:
                LogController.disable()
        
        self._local_verbose = LogController.is_enabled()

        self.issued_time = 0
        self.issued_time_lock = threading.Lock()

        self.events: List[TraceEvent] = self.load_events(event_file)
        self.event_index = 0

        # 分别记录前向和反向需要hook的模块
        self.forward_target_modules: Set[str] = set()
        self.backward_target_modules: Set[str] = set()
        self.time_to_module: Dict[int, str] = {}
        
        self.hooks = []
        self.current_module_name: Optional[str] = None
        self._probe_finished = False

        self._conditional_log_init()

    def _conditional_log_init(self):
        """条件初始化日志"""
        if self._local_verbose:
            log_message(f"[HOOK-INIT] Loaded {len(self.events)} trace events")
            log_message(f"[HOOK-INIT] Bound to device: {self.device}")

    def enable_logging(self):
        """为此实例启用日志"""
        self._local_verbose = True
        if self._verbose_override is not None:
            self._verbose_override = True

    def disable_logging(self):
        """为此实例禁用日志"""
        self._local_verbose = False
        if self._verbose_override is not None:
            self._verbose_override = False

    @property
    def is_logging_enabled(self) -> bool:
        """检查当前实例的日志状态"""
        if self._verbose_override is not None:
            return self._verbose_override
        return LogController._enabled

    def _increment_issued_time(self) -> int:
        with self.issued_time_lock:
            self.issued_time += 1
            return self.issued_time

    def reset_issued_time(self):
        with self.issued_time_lock:
            self.issued_time = 0
            self.event_index = 0

    def load_events(self, file_path: str) -> List[TraceEvent]:
        pattern = re.compile(
            r"Issued Time: (\d+) Tensor: (\d+) From: (\w+), To: (\w+) tag: (\w+)"
        )
        events = []
        with open(file_path, "r") as f:
            for line in f:
                m = pattern.match(line.strip())
                if m:
                    events.append(
                        TraceEvent(
                            int(m.group(1)),
                            int(m.group(2)),
                            m.group(3),
                            m.group(4),
                            m.group(5),
                        )
                    )
        return events

    def probe_modules(self, sample_input: torch.Tensor, sample_target: Optional[torch.Tensor] = None, criterion: Optional[nn.Module] = None) -> Tuple[Set[str], Set[str]]:
        """
        探测阶段：运行一个完整的前向+反向传播，记录每个issued_time对应的模块名称
        
        Args:
            sample_input: 用于探测的样本输入（一个batch的数据），必须在正确的device上
            sample_target: 用于计算损失的标签，如果为None则使用随机生成的target
            criterion: 损失函数，如果为None则使用CrossEntropyLoss
        
        Returns:
            (forward_target_modules, backward_target_modules): 需要hook的前向和反向模块集合
        """
        if self._probe_finished:
            if self._local_verbose:
                log_message("[HOOK-PROBE] Probe already finished, skipping")
            return self.forward_target_modules, self.backward_target_modules

        # 确保输入在正确的device上
        if sample_input.device.index != self.device:
            raise RuntimeError(
                f"Sample input device {sample_input.device} != HookManager device {self.device}"
            )

        # 如果没有提供target和criterion，创建临时的
        if sample_target is None:
            # ★ 修复：使用CPU创建tensor，然后转移到NPU，避免CUDA依赖
            sample_target = torch.randint(0, 10, (sample_input.size(0),))
            sample_target = sample_target.to(f"npu:{self.device}")
        else:
            sample_target = sample_target.to(f"npu:{self.device}")
            
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        # 临时注册全模块hook用于探测
        forward_probe_hooks = []
        backward_probe_hooks = []
        
        forward_time_to_module: Dict[int, str] = {}
        backward_time_to_module: Dict[int, str] = {}

        def make_forward_probe_hook(name):
            def probe_hook(module, inputs, output):
                issued_time = self._increment_issued_time()
                forward_time_to_module[issued_time] = name
            return probe_hook

        def make_backward_probe_hook(name):
            def probe_hook(module, grad_input, grad_output):
                issued_time = self._increment_issued_time()
                backward_time_to_module[issued_time] = name
            return probe_hook

        # 在正确的device上下文中注册临时探测hook
        with torch.npu.device(self.device):
            for name, module in self.model.named_modules():
                if isinstance(module, OP_MAP):
                    forward_probe_hooks.append(module.register_forward_hook(make_forward_probe_hook(name)))
                    backward_probe_hooks.append(module.register_full_backward_hook(make_backward_probe_hook(name)))

        if self._local_verbose:
            log_message(f"[HOOK-PROBE] Registered {len(forward_probe_hooks)} forward probe hooks")
            log_message(f"[HOOK-PROBE] Registered {len(backward_probe_hooks)} backward probe hooks")

        try:
            # 重置时间戳
            self.reset_issued_time()
            
            # 运行完整的前向+反向传播
            output = self.model(sample_input)
            loss = criterion(output, sample_target)
            loss.backward()
            
            # 分析哪些时间戳有事件
            event_times = {ev.issued_time for ev in self.events}
            
            # 找到需要前向hook的模块
            for time in event_times:
                if time in forward_time_to_module:
                    module_name = forward_time_to_module[time]
                    self.forward_target_modules.add(module_name)
                    self.time_to_module[time] = module_name
            
            # 找到需要反向hook的模块
            for time in event_times:
                if time in backward_time_to_module:
                    module_name = backward_time_to_module[time]
                    self.backward_target_modules.add(module_name)
                    self.time_to_module[time] = module_name

            if self._local_verbose:
                log_message(f"[HOOK-PROBE] Found {len(self.forward_target_modules)} modules for forward hook")
                log_message(f"[HOOK-PROBE] Found {len(self.backward_target_modules)} modules for backward hook")

        finally:
            # 清理临时hook
            for h in forward_probe_hooks:
                h.remove()
            for h in backward_probe_hooks:
                h.remove()
            
            # 重置状态
            self.reset_issued_time()
            self._probe_finished = True

        return self.forward_target_modules, self.backward_target_modules

    def register_hooks(self, model: nn.Module = None):
        """
        在探测完成后，只在需要的模块上注册对应的hook
        必须在probe_modules之后调用
        
        Args:
            model: 目标模型，默认为self.model
        """
        if not self._probe_finished:
            raise RuntimeError("Must call probe_modules() before register_hooks()")
        
        if model is None:
            model = self.model
            
        # 清理已有的hook（如果有）
        self.remove_hooks()
        
        # 只在目标模块上注册对应的hook（前向和反向分开处理）
        registered_forward = 0
        registered_backward = 0
        
        with torch.npu.device(self.device):
            for name, module in model.named_modules():
                # 注册前向hook
                if name in self.forward_target_modules:
                    self.hooks.append(module.register_forward_hook(self._make_forward_hook(name)))
                    registered_forward += 1
                
                # 注册反向hook
                if name in self.backward_target_modules:
                    self.hooks.append(module.register_full_backward_hook(self._make_backward_hook(name)))
                    registered_backward += 1

        if self._local_verbose:
            log_message(f"[HOOK-REG] Registered {registered_forward} forward hooks")
            log_message(f"[HOOK-REG] Registered {registered_backward} backward hooks")

    def setup(self, sample_input: torch.Tensor, sample_target: Optional[torch.Tensor] = None, criterion: Optional[nn.Module] = None):
        """
        便捷方法：一键完成探测和注册
        
        Args:
            sample_input: 用于探测的样本输入，必须在正确的device上
            sample_target: 用于计算损失的标签，可选
            criterion: 损失函数，可选
        """
        self.probe_modules(sample_input, sample_target, criterion)
        self.register_hooks()

    def _make_forward_hook(self, name):
        def forward_hook(module, inputs, output):
            issued_time = self._increment_issued_time()
            self.current_module_name = name
            
            if self._local_verbose:
                pass
            
            self.process_events(issued_time, inputs, output)
        return forward_hook

    def _make_backward_hook(self, name):
        def backward_hook(module, grad_input, grad_output):
            issued_time = self._increment_issued_time()
            self.process_events(issued_time)
        return backward_hook

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        
        if self._local_verbose:
            log_message("[HOOK] All hooks removed")

    def process_events(self, current_time: int, inputs=None, output=None):
        with torch.npu.device(self.device):
            compute_stream = torch_npu.npu.current_stream()
        
        if hasattr(compute_stream, 'device') and compute_stream.device.index != self.device:
            raise RuntimeError(
                f"Stream device {compute_stream.device} != HookManager device {self.device}"
            )
        
        verbose = self._local_verbose

        while self.event_index < len(self.events) and self.events[self.event_index].issued_time == current_time:
            ev = self.events[self.event_index]
            tid = ev.tensor_id

            if ev.from_location == "Not_Known" and ev.to_location == "In_gpu":
                tensor = None
                if ev.tag == "input":
                    tensor = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
                elif ev.tag == "output":
                    tensor = output
                elif ev.tag == "weight":
                    module = self.model.get_submodule(self.current_module_name)
                    tensor = self._get_module_weights(module)

                if tensor is not None:
                    if tensor.device.index != self.device:
                        raise RuntimeError(
                            f"Tensor device {tensor.device} != HookManager device {self.device}"
                        )
                    
                    self.swap_manager.add_swap_tensor(tid, tensor)
                    if verbose:
                        log_lifecycle("extract", tid, current_time)

            elif ev.from_location == "In_gpu" and ev.to_location == "In_cpu":
                self.swap_manager.launch_d2h(tid, compute_stream)
                if verbose:
                    log_lifecycle("d2h", tid, current_time)

            elif ev.from_location == "In_cpu" and ev.to_location == "In_cpu":
                self.swap_manager.wait_d2h_finished(tid, compute_stream)
                if verbose:
                    log_lifecycle("wait_d2h", tid, current_time)

            elif ev.from_location == "In_cpu" and ev.to_location == "In_gpu":
                self.swap_manager.launch_h2d(tid, compute_stream)
                if verbose:
                    log_lifecycle("h2d", tid, current_time)

            elif ev.from_location == "In_gpu" and ev.to_location == "In_gpu":
                self.swap_manager.wait_h2d_finished(tid, compute_stream)
                if verbose:
                    log_lifecycle("wait_h2d", tid, current_time)
                    if self.swap_manager.is_h2d_finished(tid):
                        log_message(f"[HOOK] H2D finished for Tensor {tid} at time {current_time}")

            self.event_index += 1

    def _get_module_weights(self, module: nn.Module) -> Optional[torch.Tensor]:
        if module is None:
            return None
        weights = []
        for name in ("weight", "bias"):
            p = getattr(module, name, None)
            if p is not None:
                weights.append(p.view(-1))
        return torch.cat(weights) if weights else None


def enable_hook_logging():
    LogController.enable()


def disable_hook_logging():
    LogController.disable()


def is_hook_logging_enabled() -> bool:
    return LogController.is_enabled()