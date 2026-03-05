import os
import atexit
import threading
import torch
import torch.nn as nn
import re
from typing import List, Optional
from functools import wraps

import torch_npu

from .swapManager import SwapManager
from .module_transfer import ReLU, Cat, Add, AvgPool2d, MaxPool2d


# =========================
# 全局日志控制配置（高性能关键：模块级常量，避免运行时查找）
# =========================
# 通过环境变量 HOOK_VERBOSE 控制，1 或 true 开启，默认关闭
_HOOK_VERBOSE = os.getenv("HOOK_VERBOSE", "0").lower() in ("1", "true", "yes", "on")

# 日志文件句柄（延迟初始化，且仅在开启日志时创建）
_LOG_F = None
_LOG_F_LOCK = threading.Lock() if _HOOK_VERBOSE else None  # 关闭时连锁都不创建

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
    """延迟初始化日志文件，仅在首次需要时打开"""
    global _LOG_F
    if _LOG_F is None:
        log_path = os.getenv("HOOK_LOG", "train_trace.log")
        _LOG_F = open(log_path, "w", buffering=1)
        atexit.register(lambda: _LOG_F and _LOG_F.close())
    return _LOG_F


# =========================
# 高性能日志装饰器（零开销抽象）
# =========================
def _conditional_log(log_func):
    """
    条件日志装饰器：当 _HOOK_VERBOSE=False 时，被装饰的函数变成空操作
    实现原理：在导入时根据配置决定函数行为，避免运行时判断
    """
    if _HOOK_VERBOSE:
        @wraps(log_func)
        def wrapper(*args, **kwargs):
            return log_func(*args, **kwargs)
        return wrapper
    else:
        # 关闭日志时：返回一个什么都不做的函数，会被 Python 优化为极低开销
        def noop(*args, **kwargs):
            pass
        # 保留函数签名信息用于调试
        noop.__name__ = log_func.__name__
        noop.__doc__ = f"[DISABLED] {log_func.__doc__ or ''}"
        return noop


# =========================
# 日志函数（使用条件装饰器）
# =========================
@_conditional_log
def log_message(message: str) -> None:
    """记录普通消息（仅在 HOOK_VERBOSE=1 时执行）"""
    with _LOG_F_LOCK:
        print(message, file=_get_log_f())


@_conditional_log
def log_lifecycle(action: str, tensor_id: int, time: int = 0) -> None:
    """记录生命周期事件（仅在 HOOK_VERBOSE=1 时执行）"""
    with _LOG_F_LOCK:
        print(
            f"[SWAP-LIFE] {action:10} | Tensor: {tensor_id:4} | Issued: {time:4}",
            file=_get_log_f()
        )


# =========================
# 运行时动态控制（供 HookManager 使用）
# =========================
class LogController:
    """
    运行时日志控制器：支持动态开关，但保持高性能
    设计为单例模式，通过类属性控制，避免实例查找开销
    """
    _enabled: bool = _HOOK_VERBOSE  # 初始值跟随环境变量
    _lock = threading.Lock() if _HOOK_VERBOSE else None
    
    @classmethod
    def enable(cls) -> None:
        """启用日志（线程安全）"""
        if not cls._lock:
            # 如果初始化时没创建锁，现在创建一个
            cls._lock = threading.Lock()
        with cls._lock:
            cls._enabled = True
    
    @classmethod
    def disable(cls) -> None:
        """禁用日志（线程安全）"""
        if cls._lock:
            with cls._lock:
                cls._enabled = False
    
    @classmethod
    def is_enabled(cls) -> bool:
        """检查日志状态（读操作无需锁，利用 GIL 保证原子性）"""
        return cls._enabled
    
    @classmethod
    def toggle(cls) -> bool:
        """切换日志状态，返回新状态"""
        if not cls._lock:
            cls._lock = threading.Lock()
        with cls._lock:
            cls._enabled = not cls._enabled
            return cls._enabled


# 高性能内联日志宏（替代函数调用）
def _fast_log_check() -> bool:
    """
    快速日志检查：单层属性查找，C 级别优化
    在 CPython 中，类属性查找经过优化，开销极低
    """
    return LogController._enabled


# =========================
# Event 描述对象（纯数据）
# =========================
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
            swap_manager: 交换管理器
            event_file: 事件文件路径
            model: 目标模型
            verbose: 是否打印日志，None 表示使用全局设置，True/False 强制覆盖
        """
        self.swap_manager = swap_manager
        self.model = model
        
        # 实例级别的日志控制（优先级高于全局设置）
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

        self.hooks = []
        self.current_module_name: Optional[str] = None

        # 初始化日志（仅在需要时）
        self._conditional_log_init()
        self.register_hooks(model)

    def _conditional_log_init(self):
        """条件初始化日志：仅在开启时记录"""
        if self._local_verbose:
            log_message(f"[HOOK-INIT] Loaded {len(self.events)} trace events")

    # ---------- 便捷的日志控制接口 ----------
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

    # ---------- issued time ----------
    def _increment_issued_time(self) -> int:
        with self.issued_time_lock:
            self.issued_time += 1
            return self.issued_time

    def reset_issued_time(self):
        with self.issued_time_lock:
            self.issued_time = 0
            self.event_index = 0

    # ---------- trace events ----------
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

    # ---------- hooks ----------
    def _make_forward_hook(self, name):
        def forward_hook(module, inputs, output):
            issued_time = self._increment_issued_time()
            self.current_module_name = name
            
            # 高性能检查：本地属性访问，避免函数调用开销
            if self._local_verbose:
                # 只有在开启日志时才记录时间等信息
                pass  # 具体的 verbose 日志可在此添加
            
            self.process_events(issued_time, inputs, output)
        return forward_hook

    def _make_backward_hook(self, name):
        def backward_hook(module, grad_input, grad_output):
            issued_time = self._increment_issued_time()
            self.process_events(issued_time)
        return backward_hook

    def register_hooks(self, model: nn.Module):
        for name, module in model.named_modules():
            if isinstance(module, OP_MAP):
                self.hooks.append(module.register_forward_hook(self._make_forward_hook(name)))
                self.hooks.append(module.register_full_backward_hook(self._make_backward_hook(name)))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        
        # 条件日志：使用局部状态检查，避免全局锁
        if self._local_verbose:
            log_message("[HOOK] All hooks removed")

    # =========================
    # 核心调度逻辑（零开销日志检查）
    # =========================
    def process_events(self, current_time: int, inputs=None, output=None):
        compute_stream = torch_npu.npu.current_stream()
        
        # 缓存本地状态，避免重复属性查找
        verbose = self._local_verbose

        while self.event_index < len(self.events) and self.events[self.event_index].issued_time == current_time:
            ev = self.events[self.event_index]
            tid = ev.tensor_id

            # ---------- Extract tensor ----------
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
                    self.swap_manager.add_swap_tensor(tid, tensor)
                    # 内联日志检查：仅在开启时调用
                    if verbose:
                        log_lifecycle("extract", tid, current_time)

            # ---------- Device to Host ----------
            elif ev.from_location == "In_gpu" and ev.to_location == "In_cpu":
                self.swap_manager.launch_d2h(tid, compute_stream)
                if verbose:
                    log_lifecycle("d2h", tid, current_time)

            # ---------- Wait D2H ----------
            elif ev.from_location == "In_cpu" and ev.to_location == "In_cpu":
                self.swap_manager.wait_d2h_finished(tid, compute_stream)
                if verbose:
                    log_lifecycle("wait_d2h", tid, current_time)

            # ---------- Host to Device ----------
            elif ev.from_location == "In_cpu" and ev.to_location == "In_gpu":
                self.swap_manager.launch_h2d(tid, compute_stream)
                if verbose:
                    log_lifecycle("h2d", tid, current_time)

            # ---------- Wait H2D ----------
            elif ev.from_location == "In_gpu" and ev.to_location == "In_gpu":
                self.swap_manager.wait_h2d_finished(tid, compute_stream)
                if verbose:
                    log_lifecycle("wait_h2d", tid, current_time)
                    if self.swap_manager.is_h2d_finished(tid):
                        log_message(f"[HOOK] H2D finished for Tensor {tid} at time {current_time}")

            self.event_index += 1

    # ---------- utils ----------
    def _get_module_weights(self, module: nn.Module) -> Optional[torch.Tensor]:
        if module is None:
            return None
        weights = []
        for name in ("weight", "bias"):
            p = getattr(module, name, None)
            if p is not None:
                weights.append(p.view(-1))
        return torch.cat(weights) if weights else None


# =========================
# 便捷的模块级接口
# =========================
def enable_hook_logging():
    """全局启用 Hook 日志"""
    LogController.enable()


def disable_hook_logging():
    """全局禁用 Hook 日志"""
    LogController.disable()


def is_hook_logging_enabled() -> bool:
    """查询全局日志状态"""
    return LogController.is_enabled()

