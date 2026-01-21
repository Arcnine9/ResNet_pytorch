import os
import atexit
import threading
import torch
import torch.nn as nn
import re
from typing import List,Optional

import torch_npu

from .swapManager import SwapManager
from .module_transfer import ReLU, Cat, Add, AvgPool2d, MaxPool2d


# =========================
# 日志
# =========================
_LOG_F = None
_LOG_F_LOCK = threading.Lock()

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
        print(
            f"[SWAP-LIFE] {action:10} | Tensor: {tensor_id:4} | Issued: {time:4}",
            file=_get_log_f()
        )


# =========================
# Event 描述对象（纯数据）
# =========================
class TraceEvent:
    def __init__(self, issued_time: int, tensor_id: int, from_location: str, to_location: str, tag: str):
        self.issued_time = issued_time
        self.tensor_id = tensor_id
        self.from_location = from_location
        self.to_location = to_location
        self.tag = tag


class HookManager:
    def __init__(self, swap_manager: SwapManager, event_file: str, model: nn.Module):
        self.swap_manager = swap_manager
        self.model = model

        self.issued_time = 0
        self.issued_time_lock = threading.Lock()

        self.events: List[TraceEvent] = self.load_events(event_file)
        self.event_index = 0

        self.hooks = []
        self.current_module_name: Optional[str] = None

        log_message(f"[HOOK-INIT] Loaded {len(self.events)} trace events")
        self.register_hooks(model)

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
        log_message("[HOOK] All hooks removed")

    # =========================
    # 核心调度逻辑
    # =========================
    def process_events(self, current_time: int, inputs=None, output=None):
        compute_stream = torch_npu.npu.current_stream()

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
                    log_lifecycle("extract", tid, current_time)

            # ---------- Device to Host ----------
            elif ev.from_location == "In_gpu" and ev.to_location == "In_cpu":
                self.swap_manager.launch_d2h(tid, compute_stream)
                log_lifecycle("d2h", tid, current_time)

            # ---------- Wait D2H ----------
            elif ev.from_location == "In_cpu" and ev.to_location == "In_cpu":
                self.swap_manager.wait_d2h_finished(tid, compute_stream)
                log_lifecycle("wait_d2h", tid, current_time)

            # ---------- Host to Device ----------
            elif ev.from_location == "In_cpu" and ev.to_location == "In_gpu":
                self.swap_manager.launch_h2d(tid, compute_stream)
                log_lifecycle("h2d", tid, current_time)

            # ---------- Wait H2D ----------
            elif ev.from_location == "In_gpu" and ev.to_location == "In_gpu":
                self.swap_manager.wait_h2d_finished(tid, compute_stream)
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
