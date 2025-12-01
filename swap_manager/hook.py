import torch
import torch_npu

# 全局层计数器（仅用于打印序号，可不要）
_layer_idx = 0

def _make_print_hook(module_name, is_forward):
    """
    返回一个 hook 函数，打印：
     - 层名
     - 前向/反向
     - 输入/输出 shape
     - 当前 device
     - NPU 已用显存
    """
    def hook_fn(module, input, output=None):
        global _layer_idx
        io = "FWD" if is_forward else "BWD"
        # 取第一个 tensor 代表输入/输出
        inp = input[0] if isinstance(input, (tuple, list)) else input
        out = output[0] if isinstance(output, (tuple, list)) else output
        print(f"[{io}] #{_layer_idx:03d} | {module_name:30s} | "
              f"inp={inp.shape} dev={inp.device} | "
              f"out={out.shape if out is not None else 'None'} | ")
        if is_forward:
            _layer_idx += 1
    return hook_fn

def register_all_hooks(model, verbose=True):
    """
    递归给 model 的每一层注册
    forward hook  +  backward hook（tensor 级）
    """
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:      # 只挂叶子层，可去掉条件挂全部
            continue
        # 1) 前向钩子
        module.register_forward_hook(_make_print_hook(name, is_forward=True))
        # 2) 反向钩子（tensor 级）→ 对每层输出 tensor 注册
        def _tensor_hook_factory(name):
            def tensor_hook(grad):
                print(f"[BWD] {name:30s} | grad_shape={grad.shape} "
                      f"grad_mean={grad.mean().item():.5f}")
            return tensor_hook
        # 把 tensor 级钩子挂在 module 的输出上
        def forward_hook_for_tensor(module, input, output):
            if isinstance(output, torch.Tensor):
                output.register_hook(_tensor_hook_factory(name))
        module.register_forward_hook(forward_hook_for_tensor)