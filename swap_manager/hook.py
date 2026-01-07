import os
import atexit
import threading
import torch
import torch.nn as nn
from . import module_transfer as fx
# ---------------- 日志文件句柄 ----------------
_LOG_F = None
_LOG_F_LOCK = threading.Lock()

OP_MAP = (    nn.Conv2d,
    nn.ReLU, nn.ReLU6, nn.LeakyReLU,
    nn.MaxPool2d,
    nn.AdaptiveAvgPool2d,
    nn.Linear,
    nn.Dropout, nn.Dropout2d,
    nn.BatchNorm2d,
    fx.Cat, fx.Add, fx.ReLU, fx.AvgPool2d, fx.MaxPool2d,
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

# ---------------- 钩子管理器 ----------------
class HookManager:
    def __init__(self):
        self.issued_time = 0
        self.issued_time_lock = threading.Lock()
        self.hooks = []

    def _increment_issued_time(self):
        with self.issued_time_lock:
            self.issued_time += 1
            return self.issued_time
    
    def reset_issued_time(self):
        with self.issued_time_lock:
            self.issued_time = 0

    def _make_forward_hook(self, name, module):
        def forward_hook(module, input, output):
            issued_time = self._increment_issued_time()
            log_message(f"[FWD-END] Issued Time: {issued_time}, Layer: {name}")
        return forward_hook

    def _make_backward_hook(self, name, module):
        def backward_hook(module, grad_input, grad_output):
            issued_time = self._increment_issued_time()
            log_message(f"[BWD-BEGIN] Issued Time: {issued_time}, Layer: {name}")
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

# ---------------- 对外接口 ----------------
def register_all_hooks(model):
    """遍历模型，给所有支持的模块挂探针"""
    hook_manager = HookManager()
    hook_manager.register_hooks(model)
    return hook_manager