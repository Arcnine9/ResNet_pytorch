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

# ---------- FX 替换核心 ----------
def replace_functional(model: nn.Module,
                       verbose: bool = False) -> nn.Module:
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

        elif node.target == torch.add:
            add, name = Add(), f"add_{node.name}"
            traced.add_module(name, add)
            with traced.graph.inserting_before(node):
                new_node = traced.graph.call_module(name, args=node.args, kwargs=None)
            node.replace_all_uses_with(new_node)
            traced.graph.erase_node(node)
            if verbose:
                print(f"[FX] Replaced torch.add at {node.name}")

    traced.recompile()
    return traced