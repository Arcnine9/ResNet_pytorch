# hook.py
# 简化版：移除层变化检测，直接按事件处理

import os
import atexit
import threading
import torch
import torch.nn as nn
import re
from typing import List, Optional

import torch_npu

from .swap_manager import SwapManager
from .module_transfer import ReLU, Cat, Add, AvgPool2d, MaxPool2d


# ========================= 日志 =========================
_LOG_F = None
_LOG_F_LOCK = threading.Lock()

OP_MAP = (
    nn.Conv2d, nn.ReLU, nn.ReLU6, nn.LeakyReLU,
    nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.Linear,
    nn.Dropout, nn.Dropout2d, nn.BatchNorm2d,
    ReLU, Cat, Add, AvgPool2d, MaxPool2d,
)


def _get_log_f():
    global _LOG_F
    if _LOG_F is None:
        log_path = os.getenv("HOOK_LOG", "train_trace.log")
        _LOG_F = open(log_path, "w", buffering=1)
        atexit.register(lambda: _LOG_F and _LOG_F.close())
    return _LOG_F


def log_message(message):
    with _LOG_F_LOCK:
        print(message, file=_get_log_f())


def log_lifecycle(action: str, tensor_id: int, time: int = 0):
    with _LOG_F_LOCK:
        print(f"[{action:12}] tensor={tensor_id:4} time={time:4}", file=_get_log_f())


class TraceEvent:
    def __init__(self, issued_time: int, tensor_id: int, from_loc: str, to_loc: str, tag: str):
        self.issued_time = issued_time
        self.tensor_id = tensor_id
        self.from_location = from_loc
        self.to_location = to_loc
        self.tag = tag


class HookManager:
    def __init__(self, swap_manager: SwapManager, event_file: str, model: nn.Module, device=None):
        self.swap_manager = swap_manager
        self.model = model
        self.device = device if device is not None else torch_npu.npu.current_device()
        
        self.issued_time = 0
        self.issued_time_lock = threading.Lock()
        self.events: List[TraceEvent] = self.load_events(event_file)
        self.event_index = 0
        self.hooks = []
        self.current_module_name = None
        
        print(f"[HookManager] {len(self.events)} events, device={self.device}")
        self.register_hooks(model)

    def _increment_issued_time(self) -> int:
        with self.issued_time_lock:
            self.issued_time += 1
            return self.issued_time

    def reset_issued_time(self):
        with self.issued_time_lock:
            self.issued_time = 0
            self.event_index = 0

    def load_events(self, file_path: str) -> List[TraceEvent]:
        pattern = re.compile(r"Issued Time: (\d+) Tensor: (\d+) From: (\w+), To: (\w+) tag: (\w+)")
        events = []
        with open(file_path, "r") as f:
            for line in f:
                m = pattern.match(line.strip())
                if m:
                    events.append(TraceEvent(int(m.group(1)), int(m.group(2)),
                                           m.group(3), m.group(4), m.group(5)))
        return events

    def register_hooks(self, model: nn.Module):
        for name, module in model.named_modules():
            if isinstance(module, OP_MAP):
                self.hooks.append(module.register_forward_hook(self._make_forward_hook(name)))
                self.hooks.append(module.register_full_backward_hook(self._make_backward_hook(name)))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def _make_forward_hook(self, name):
        def forward_hook(module, inputs, output):
            issued_time = self._increment_issued_time()
            self.current_module_name = name
            self.process_events(issued_time, inputs, output)
        return forward_hook

    def _make_backward_hook(self, name):
        def backward_hook(module, grad_input, grad_output):
            issued_time = self._increment_issued_time()
            self.process_events(issued_time, None, None)
        return backward_hook

    # ========================= 核心：简化的事件处理 =========================
    
    def process_events(self, current_time: int, inputs, output):
        """
        简化的事件处理：
        - 每个事件独立处理，不依赖层状态
        - 直接映射到 SwapManager 的同步接口
        """
        with torch.npu.device(self.device):
            _ = torch_npu.npu.current_stream()

        while self.event_index < len(self.events) and self.events[self.event_index].issued_time == current_time:
            ev = self.events[self.event_index]
            tid = ev.tensor_id

            # Extract & Wrap
            if ev.from_location == "Not_Known" and ev.to_location == "In_gpu":
                tensor = self._extract_tensor(ev.tag, inputs, output)
                if tensor is not None:
                    wrapped = self.swap_manager.wrap_tensor(tid, tensor)
                    log_lifecycle("wrap", tid, current_time)
                    print(f"[{current_time}] wrap {tid}")

            # D2H（同步）
            elif ev.from_location == "In_gpu" and ev.to_location == "In_cpu":
                # 如果 trace 设计是立即执行，可以直接调用 execute_d2h
                success = self.swap_manager.execute_d2h(tid)
                log_lifecycle("execute_d2h", tid, current_time)
                print(f"[{current_time}] execute_d2h {tid}, success={success}")

            # Execute D2H（同步）
            elif ev.from_location == "In_cpu" and ev.to_location == "In_cpu":
                log_lifecycle("host check", tid, current_time)
                print(f"[{current_time}] host check {tid}")

            # Execute H2D（同步）
            elif ev.from_location == "In_cpu" and ev.to_location == "In_gpu":
                success = self.swap_manager.execute_h2d(tid)
                log_lifecycle("execute_h2d", tid, current_time)
                print(f"[{current_time}] execute_h2d {tid}, success={success}")

            # Check on device
            elif ev.from_location == "In_gpu" and ev.to_location == "In_gpu":
                success = self.swap_manager.check_on_device(tid)
                log_lifecycle("check_device", tid, current_time)
                print(f"[{current_time}] check_device {tid}, success={success}")

            self.event_index += 1

    def _extract_tensor(self, tag: str, inputs, output) -> Optional[torch.Tensor]:
        if tag == "input":
            if isinstance(inputs, (list, tuple)) and len(inputs) > 0:
                return inputs[0]
            return inputs
        elif tag == "output":
            return output
        elif tag == "weight":
            module = self.model.get_submodule(self.current_module_name)
            return self._get_module_weights(module)
        return None

    def _get_module_weights(self, module: nn.Module) -> Optional[torch.Tensor]:
        if module is None:
            return None
        weights = []
        for name in ("weight", "bias"):
            p = getattr(module, name, None)
            if p is not None:
                weights.append(p.view(-1))
        return torch.cat(weights) if weights else None