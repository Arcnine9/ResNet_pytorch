# hooks.py
import os
import atexit
import torch
import torch.nn as nn
from . import module_transfer as fx

# ---------------- 日志文件句柄 ----------------
_LOG_F = None
_LAYER_ID = 0          # 全局层序号
_printTensor = 0 # 0 print layer info only, 1 print tensor info

OP_MAP = {
    nn.Conv2d,
    nn.ReLU, nn.ReLU6, nn.LeakyReLU,
    nn.MaxPool2d,
    nn.AdaptiveAvgPool2d,
    nn.Linear,
    nn.Dropout, nn.Dropout2d,
    nn.BatchNorm2d,
    fx.Cat, fx.Add, fx.ReLU,
}

def _get_log_f():
    global _LOG_F
    if _LOG_F is None:
        log_path = os.getenv("HOOK_LOG", "train_trace.log")
        _LOG_F = open(log_path, "w", buffering=1)
        atexit.register(lambda: _LOG_F and _LOG_F.close())
    return _LOG_F


# ---------------- 单模块探针 ----------------
def make_probe_hook(name, mod):
    global _LAYER_ID
    
    def _print_layer_info(layer_id, tag, layer_name):
        log = f"[{tag}] #{layer_id:03d} | {layer_name:40s}"
        print(log, file=_get_log_f())

    def _print_tensor(layer_id, tag, t, idx=None, ):
        if t is None or not isinstance(t, torch.Tensor):
            return
        idx_str = f"[{idx}]" if idx is not None else ""
        log = (f"[{tag}] #{layer_id:03d} | {name:40s} | "
               f"{idx_str}shape={tuple(t.shape)} ptr={t.data_ptr()} "
               f"size={t.numel() * t.element_size()}B")
        print(log, file=_get_log_f())

    # ===== 前向 pre：输入 + 参数 =====
    def pre_hook(m, inp):
        global _LAYER_ID
        layer_id = _LAYER_ID
        # 1. 所有输入（含 Cat 的 list）
        if _printTensor == 1:
            if isinstance(m, fx.Cat):
                # Cat 的多输入在 pos=0 的 list 里
                for pos, item in enumerate(inp if isinstance(inp, (list, tuple)) else [inp]):
                    if isinstance(item, list):
                        for idx, t in enumerate(item):
                            _print_tensor(layer_id, "FWD-PRE", t, f"inp[{pos}][{idx}]")
                    elif isinstance(item, torch.Tensor):
                        _print_tensor(layer_id, "FWD-PRE", item, f"inp[{pos}]")
            else:
                for idx, t in enumerate(inp if isinstance(inp, (list, tuple)) else [inp]):
                    _print_tensor(layer_id, "FWD-PRE", t, f"inp[{idx}]")

            # 2. 参数 & 统计量（仅含参层）
            for p_name in ['weight', 'bias']:
                param = getattr(m, p_name, None)
                _print_tensor(layer_id, "FWD-PRE", param, p_name)
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                for stat_name in ['running_mean', 'running_var']:
                    stat = getattr(m, stat_name, None)
                    _print_tensor(layer_id, "FWD-PRE", stat, stat_name)
        elif _printTensor == 0:
            _print_layer_info(layer_id, "FWD-PRE", name)

    # ===== 后向 post：输出 + 所有梯度 =====
    def post_hook(m, inp, out):
        global _LAYER_ID
        layer_id = _LAYER_ID  # 与 pre_hook 保持一致
        _LAYER_ID += 1
        # 1. 所有输出
        if _printTensor == 1:
            if isinstance(m, fx.Cat):
                for pos, item in enumerate(out if isinstance(out, (list, tuple)) else [out]):
                    if isinstance(item, list):
                        for idx, t in enumerate(item):
                            _print_tensor(layer_id, "FWD-POST", t, f"out[{pos}][{idx}]")
                    elif isinstance(item, torch.Tensor):
                        _print_tensor(layer_id, "FWD-POST", item, f"out[{pos}]")
            else:
                for idx, t in enumerate(out if isinstance(out, (list, tuple)) else [out]):
                    _print_tensor(layer_id, "FWD-POST", t, f"out[{idx}]")

            # 2. 所有梯度（d_input / d_output / d_weight / d_bias）
            if isinstance(m, fx.Cat):
                # d_input 在 inp 里（反向时 inp 就是 grad）
                for pos, item in enumerate(inp if isinstance(inp, (list, tuple)) else [inp]):
                    if isinstance(item, list):
                        for idx, g in enumerate(item):
                            _print_tensor(layer_id, "FWD-POST", g, f"d_in[{pos}][{idx}]")
                    elif isinstance(item, torch.Tensor):
                        _print_tensor(layer_id, "FWD-POST", item, f"d_in[{pos}]")
            else:
                for idx, g in enumerate(inp if isinstance(inp, (list, tuple)) else [inp]):
                    _print_tensor(layer_id, "FWD-POST", g, f"d_in[{idx}]")

            # d_weight / d_bias
            for p_name in ['weight', 'bias']:
                if hasattr(m, p_name) and getattr(m, p_name) is not None:
                    g = getattr(m, p_name).grad
                    _print_tensor(layer_id, "FWD-POST", g, f"d_{p_name}")
        elif _printTensor == 0:
            _print_layer_info(layer_id, "FWD-POST", name)

        # d_output 就是 out 的 grad（注册在 tensor 上）
        for idx, t in enumerate(out if isinstance(out, (list, tuple)) else [out]):
            if isinstance(t, torch.Tensor) and t.requires_grad:
                if _printTensor == 1:
                    def grad_hook(grad, idx_=idx):
                        _print_tensor(layer_id, "BWD", grad, f"d_out[{idx_}]")
                    t.register_hook(grad_hook)
                elif _printTensor == 0:
                    def grad_pre_hook(grad):
                        _print_layer_info(layer_id, "BWD-PRE", name)
                        return grad
                    t.register_hook(grad_pre_hook)
        
        if _printTensor == 0:
            for idx, t in enumerate(inp if isinstance(inp, (list, tuple)) else [inp]):
                if isinstance(t, torch.Tensor) and t.requires_grad:
                    # 输入张量的第一个钩子（空钩子）
                    def grad_empty_hook(grad):
                        return grad

                    # 输入张量的第二个钩子（实际操作）
                    def grad_post_hook(grad):
                        _print_layer_info(layer_id, "BWD-POST", name)
                        return grad

                    # 注册两个钩子
                    t.register_hook(grad_empty_hook)  # 空钩子
                    t.register_hook(grad_post_hook)  # 实际操作的钩子


    mod.register_forward_pre_hook(pre_hook)
    mod.register_forward_hook(post_hook)

# ---------------- 对外接口 ----------------
def register_all_hooks(model):
    """遍历模型，给所有 OP_MAP 里的模块挂探针"""
    for name, module in model.named_modules():
        if type(module) not in OP_MAP or "downsample" in name:
            continue
        make_probe_hook(name, module)