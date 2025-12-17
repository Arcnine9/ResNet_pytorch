"""
module_transfer.py
专职：通过 torch.FX 将模型中 functional 的 relu/cat/add 替换为可 hook 的 nn.Module
不care Hook，不care训练/推理，只输出可trace的GraphModule
使用示例：
# train.py 或任意脚本
from torchvision.models import inception_v3
from module_transfer import replace_functional

net = inception_v3(aux_logits=False, init_weights=True)
net = replace_functional(net)   # ← 只做替换
net = net.to(device)

# 后续 Hook 由你的 hook_manager 统一处理
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
from typing import Optional, Union, List, Dict, Any

__all__ = ['replace_functional', 'ReLU', 'Cat', 'Add']

import os, atexit
_LOG_FX = None

def _get_fx_log():
    global _LOG_FX
    if _LOG_FX is None:
        log_path = os.getenv("FX_DEBUG_LOG", "fx_debug.log")
        _LOG_FX = open(log_path, "w", buffering=1)
        atexit.register(lambda: _LOG_FX and _LOG_FX.close())
    return _LOG_FX

# ---------- 可 hook 的 Module ----------
class ReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x, inplace=False)   # 禁止 inplace，方便 hook

class Cat(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, *xs: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        flat = []
        for x in xs:
            if isinstance(x, (list, tuple)):
                flat.extend([t for t in x if isinstance(t, torch.Tensor)])
            elif isinstance(x, torch.Tensor):
                flat.append(x)
        flat = [t.contiguous() for t in flat]
        return torch.cat(flat, dim=self.dim)

class Add(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y

# ---------- 可 hook 的 Module ----------
class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: Union[int, tuple], stride: Optional[Union[int, tuple]] = None, padding: Union[int, tuple] = 0, dilation: Union[int, tuple] = 1, ceil_mode: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(x, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode)


class AvgPool2d(nn.Module):
    def __init__(self, kernel_size: Union[int, tuple], stride: Optional[Union[int, tuple]] = None, padding: Union[int, tuple] = 0):
        super().__init__()
        self.avgpool = nn.AvgPool2d(kernel_size, stride, padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.avgpool(x)


# ---------- FX 替换核心 ----------
def replace_functional(model: nn.Module, verbose: bool = False) -> nn.Module:
    """
    返回：GraphModule（也是 nn.Module），所有 F.relu/torch.cat/torch.add 已被 Module 替换
    """
    model.eval()
    traced = fx.symbolic_trace(model)
    for node in list(traced.graph.nodes):
        if node.op != "call_function":
            continue
        if node.target in (F.relu, torch.relu):
            relu, name = ReLU(), f"relu_{node.name}"
            traced.add_module(name, relu)
            with traced.graph.inserting_before(node):
                new_node = traced.graph.call_module(name, args=node.args)
            node.replace_all_uses_with(new_node)
            traced.graph.erase_node(node)
            if verbose:
                print(f"[FX] Replaced F.relu at {node.name}")

        elif node.target == torch.cat:
            # 支持位置参数或关键字参数
            if node.args and len(node.args) >= 2 and "dim" not in node.kwargs:
                dim = node.args[1]
            else:
                dim = node.kwargs.get("dim", 1)
            cat, name = Cat(dim=dim), f"cat_{node.name}"
            traced.add_module(name, cat)
            with traced.graph.inserting_before(node):
                new_node = traced.graph.call_module(name, args=node.args)
            node.replace_all_uses_with(new_node)
            traced.graph.erase_node(node)
            if verbose:
                print(f"[FX] Replaced torch.cat at {node.name}  dim={dim}")

        elif "add" in str(node.target):
            # 确保参数是 tensor（排除 int/float）
            if all(isinstance(arg, fx.Node) for arg in node.args):
                # ===== 立即调试：确认进入分支 =====
                add, name = Add(), f"add_{node.name}"
                traced.add_module(name, add)
                with traced.graph.inserting_before(node):
                    new_node = traced.graph.call_module(name, args=node.args)
                node.replace_all_uses_with(new_node)
                traced.graph.erase_node(node)
                if verbose:
                    print(f"[FX] Replaced built-in add at {node.name}")
        
        elif node.target == F.max_pool2d:
            # 获取 max_pool2d 的参数
            kernel_size = node.args[1] if len(node.args) > 1 else node.kwargs.get("kernel_size")
            stride = node.args[2] if len(node.args) > 2 else node.kwargs.get("stride")
            # 打印调试信息
            print(f"[FX] Replaced F.max_pool2d at {node.name} with args: kernel_size={kernel_size}, stride={stride}")

            # 创建 MaxPool2d 模块
            maxpool, name = MaxPool2d(kernel_size, stride), f"maxpool_{node.name}"
            traced.add_module(name, maxpool)

            # 替换节点
            with traced.graph.inserting_before(node):
                new_node = traced.graph.call_module(name, args=(node.args[0],))
            node.replace_all_uses_with(new_node)
            traced.graph.erase_node(node)


            if verbose:
                print(f"[FX] Replaced F.max_pool2d at {node.name}")

        elif node.target == F.avg_pool2d:
            # 获取 avg_pool2d 的参数
            kernel_size = node.args[1] if len(node.args) > 1 else node.kwargs.get("kernel_size")
            stride = node.args[2] if len(node.args) > 2 else node.kwargs.get("stride")
            padding = node.args[3] if len(node.args) > 3 else node.kwargs.get("padding", 0)
            avgpool, name = AvgPool2d(kernel_size, stride, padding), f"avgpool_{node.name}"
            traced.add_module(name, avgpool)
            with traced.graph.inserting_before(node):
                new_node = traced.graph.call_module(name, args=node.args)
            node.replace_all_uses_with(new_node)
            traced.graph.erase_node(node)
            if verbose:
                print(f"[FX] Replaced F.avg_pool2d at {node.name}")

    traced.recompile()

    # for node in traced.graph.nodes:
    #     print(f"[DBG] 进入 call_function → {node.name}")
    #     if "add" in str(node.target):
    #         print(f"[DBG] 图里 add → {node.name} | target={node.target}")
    
    # ===== 零文件验证：直接打印统计 =====
    add_cnt = sum(1 for m in traced.modules() if m.__class__.__name__ == 'Add')
    cat_cnt = sum(1 for m in traced.modules() if m.__class__.__name__ == 'Cat')
    relu_cnt = sum(1 for m in traced.modules() if m.__class__.__name__ == 'ReLU')
    maxpool_cnt = sum(1 for m in traced.modules() if m.__class__.__name__ == 'MaxPool2d')
    avgpool_cnt = sum(1 for m in traced.modules() if m.__class__.__name__ == 'AvgPool2d')
    print(f"[FX] 替换完成：Add={add_cnt}, Cat={cat_cnt}, ReLU={relu_cnt}, MaxPool2d={maxpool_cnt}, AvgPool2d={avgpool_cnt}")
    return traced