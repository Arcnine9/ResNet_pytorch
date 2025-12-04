# hooks.py
import os
import atexit
import torch
import torch.nn as nn

# ---------------- 日志文件句柄 ----------------
_LOG_F = None
_LAYER_ID = 0          # 全局层序号

OP_MAP = {
    nn.Conv2d,
    nn.ReLU, nn.ReLU6, nn.LeakyReLU,
    nn.MaxPool2d,
    nn.AdaptiveAvgPool2d,
    nn.Linear,
    nn.Dropout, nn.Dropout2d,
    nn.BatchNorm2d,
    nn.AvgPool2d
}

def _get_log_f():
    global _LOG_F
    if _LOG_F is None:
        log_path = os.getenv("HOOK_LOG", "train_trace.log")
        _LOG_F = open(log_path, "w", buffering=1)
        atexit.register(lambda: _LOG_F and _LOG_F.close())
    return _LOG_F

# ---------------- tensor 统一字符串 ----------------
def _tensor_str(t):
    if t is None:
        return "None"
    return (f"shape={tuple(t.shape)} ptr={t.data_ptr()} "
            f"size={t.numel() * t.element_size()}B")

# ---------------- 单模块探针 ----------------
def make_probe_hook(name, mod):
    global _LAYER_ID
    layer_id = _LAYER_ID
    _LAYER_ID += 1

    # ===== 前向 pre =====
    def pre_hook(m, inp):
        t_in = inp[0] if isinstance(inp, (tuple, list)) else inp
        log = f"[FWD-PRE] #{layer_id:03d} | {name:40s} | inp={_tensor_str(t_in)}"
        print(log, file=_get_log_f())
        # print(log)          # 如需终端并行打印，取消注释

        # weight / bias
        for p_name in ['weight', 'bias']:
            param = getattr(m, p_name, None)
            if param is not None:
                log = f"[FWD-PRE] #{layer_id:03d} | {name:40s} | {p_name}={_tensor_str(param)}"
                print(log, file=_get_log_f())

        # BN 统计量
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            for stat_name in ['running_mean', 'running_var']:
                stat = getattr(m, stat_name, None)
                if stat is not None:
                    log = f"[FWD-PRE] #{layer_id:03d} | {name:40s} | {stat_name}={_tensor_str(stat)}"
                    print(log, file=_get_log_f())

    # ===== 前向 post + 梯度钩子 =====
    def post_hook(m, inp, out):
        t_out = out[0] if isinstance(out, (tuple, list)) else out
        log = f"[FWD-POST] #{layer_id:03d} | {name:40s} | out={_tensor_str(t_out)}"
        print(log, file=_get_log_f())

        # 给输出 tensor 挂梯度钩子
        if isinstance(t_out, torch.Tensor) and t_out.requires_grad:
            def grad_hook(grad):
                log = (f"[BWD] #{layer_id:03d} | {name:40s} | "
                       f"grad_shape={tuple(grad.shape)} "
                       f"grad_mean={grad.mean().item():.6f}")
                print(log, file=_get_log_f())
            t_out.register_hook(grad_hook)

    mod.register_forward_pre_hook(pre_hook)
    mod.register_forward_hook(post_hook)

# ---------------- 对外接口 ----------------
def register_all_hooks(model):
    """遍历模型，给所有 OP_MAP 里的模块挂探针"""
    for name, module in model.named_modules():
        if type(module) not in OP_MAP:
            continue
        make_probe_hook(name, module)