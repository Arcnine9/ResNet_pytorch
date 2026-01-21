import os
import atexit
import threading
import torch
import torch.nn as nn
from .swap_manager import SwapManager, Event
from .module_transfer import ReLU, Cat, Add, AvgPool2d, MaxPool2d
import re
from typing import List, Dict
import torch_npu
from torch_npu.contrib import transfer_to_npu


# 日志文件句柄
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

# 钩子管理器
class HookManager:
    def __init__(self, swap_manager: SwapManager, event_file: str, model: nn.Module):
        self.issued_time = 0
        self.issued_time_lock = threading.Lock()
        self.hooks = []
        self.swap_manager = swap_manager
        self.events = self.load_events(event_file)
        self.event_index = 0
        self.model = model
        self.current_module_name = None

    def _increment_issued_time(self):
        with self.issued_time_lock:
            self.issued_time += 1
            return self.issued_time
    
    def reset_issued_time(self):
        with self.issued_time_lock:
            self.issued_time = 0

    def load_events(self, file_path: str) -> List[Event]:
        """从文件中加载事件列表"""
        pattern = re.compile(r"Issued Time: (\d+) Tensor: (\d+) From: (\w+), To: (\w+) tag: (\w+)")
        events = []
        with open(file_path, 'r') as file:
            for line in file:
                match = pattern.match(line.strip())
                if match:
                    issued_time = int(match.group(1))
                    tensor_id = int(match.group(2))
                    from_location = match.group(3)
                    to_location = match.group(4)
                    tag = match.group(5)
                    event = Event(issued_time, tensor_id, from_location, to_location, tag)
                    events.append(event)
        return events

    def _make_forward_hook(self, name, module):
        def forward_hook(module, input, output):
            issued_time = self._increment_issued_time()
            log_message(f"[FWD-END] Issued Time: {issued_time}, Layer: {name}")
            # 设置当前模块名称
            self.current_module_name = name
            # 检查并触发事件
            self.process_events(issued_time, input, output)
        return forward_hook

    def _make_backward_hook(self, name, module):
        def backward_hook(module, grad_input, grad_output):
            issued_time = self._increment_issued_time()
            log_message(f"[BWD-BEGIN] Issued Time: {issued_time}, Layer: {name}")
            # 检查并触发事件
            self.process_events(issued_time)
        return backward_hook

    def register_hooks(self, model):
        for name, module in model.named_modules():
            if isinstance(module, OP_MAP):
                forward_hook = self._make_forward_hook(name, module)
                backward_hook = self._make_backward_hook(name, module)
                self.hooks.append(module.register_forward_hook(forward_hook))
                self.hooks.append(module.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        log_message("All hooks removed.")

    def process_events(self, current_time: int, input=None, output=None):
        """根据当前时间戳触发事件"""
        while self.event_index < len(self.events) and self.events[self.event_index].issued_time == current_time:
            event = self.events[self.event_index]
            tensor_id = event.tensor_id
            from_location = event.from_location
            to_location = event.to_location
            tag = event.tag

            if from_location == "Not_Known" and to_location == "In_gpu":
                # 根据 tag 获取当前模块的张量
                if tag == "input":
                    tensor = input
                elif tag == "output":
                    tensor = output
                elif tag == "weight":
                    module = self.model.get_submodule(self.current_module_name)
                    if module is None:
                        raise ValueError(f"Module {self.current_module_name} not found in the model.")
                    tensor = self.get_module_weights(module)
                else:
                    raise ValueError(f"Unsupported tag: {tag}")

                if tensor is not None:
                    # 初始化 SwapTensor
                    self.swap_manager.add_swap_tensor(tensor_id, tensor)

            elif from_location == "In_gpu" and to_location == "In_cpu":
                # 触发 device to host 操作
                self.swap_manager.launch_d2h(tensor_id, torch_npu.npu.current_stream())

            elif from_location == "In_cpu" and to_location == "In_gpu":
                # 触发 host to device 操作
                self.swap_manager.launch_h2d(tensor_id, torch_npu.npu.current_stream(), flag=True)

            self.event_index += 1

    def get_tensor_from_module(self, tag: str) -> torch.Tensor:
        """从当前模块中获取对应的 tensor"""
        if tag == "input":
            # 获取模块的输入
            return self.current_module_tensors.get('input', None)
        elif tag == "output":
            # 获取模块的输出
            return self.current_module_tensors.get('output', None)
        elif tag == "weight":
            # 获取模块的权重
            module = self.model.get_submodule(self.current_module_name)
            if module is None:
                raise ValueError(f"Module {self.current_module_name} not found in the model.")
            return self.get_module_weights(module)
        else:
            raise ValueError(f"Unsupported tag: {tag}")

    def get_module_weights(self, module: nn.Module) -> torch.Tensor:
        #TODO: Not implement, soon maybe implemented for global weights?
        """获取模块的权重"""
        weights = []
        for p_name in ['weight', 'bias']:
            param = getattr(module, p_name, None)
            if param is not None:
                weights.append(param)
        return torch.cat([w.view(-1) for w in weights]) if weights else None