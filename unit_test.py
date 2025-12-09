#!/usr/bin/env python
# -*- encoding:utf-8 -*-
"""
NPU 单元测试：验证 FX 能把 InceptionV3 的 functional 操作替换成可 hook 的 nn.Module
执行：
    python test_inception_fx_replace_npu.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3
import torch.fx as fx

# -------------- NPU 环境初始化（同训练脚本） --------------
import torch_npu
from torch_npu.contrib import transfer_to_npu   # 自动把 cuda 接口 remap 到 npu

device = torch.device("npu:0")


# -------------- 可 hook 的 Module（同你之前代码） --------------
class ReLU(nn.Module):
    def forward(self, x):                # 禁止 inplace，方便抓张量
        return F.relu(x, inplace=False)


class Cat(nn.Module):
    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = dim

    def forward(self, *xs):
        # 展平：遇到 list/tuple 就 extend，遇到 tensor 就 append
        flat = []
        for x in xs:
            if isinstance(x, (list, tuple)):
                flat.extend([t for t in x if isinstance(t, torch.Tensor)])
            elif isinstance(x, torch.Tensor):
                flat.append(x)
        # ---- 断言 ----
        flat = [t.contiguous() for t in flat]
        return torch.cat(flat, dim=self.dim)


class Add(nn.Module):
    def forward(self, x, y):
        return x + y


# -------------- FX 替换逻辑（同你之前代码） --------------
def make_inception_hookable(model):
    model.eval()
    dummy = torch.randn(1, 3, 299, 299).to(device)
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

        elif node.target == torch.cat:
            dim = node.kwargs.get("dim", 1)
            cat, name = Cat(dim=dim), f"cat_{node.name}"
            traced.add_module(name, cat)
            with traced.graph.inserting_before(node):
                new_node = traced.graph.call_module(name, args=node.args)   # 不传 kwargs
            node.replace_all_uses_with(new_node)
            traced.graph.erase_node(node)

        elif node.target == torch.add:
            add, name = Add(), f"add_{node.name}"
            traced.add_module(name, add)
            with traced.graph.inserting_before(node):
                new_node = traced.graph.call_module(name, args=node.args, kwargs=None)
            node.replace_all_uses_with(new_node)
            traced.graph.erase_node(node)

    traced.recompile()
    return traced


# -------------- 单元测试 --------------
def test_fx_replace():
    print("🔧  Loading InceptionV3 ...")
    # 复用训练脚本同款初始化，aux_logits=False 省掉分支
    net = inception_v3(aux_logits=False, init_weights=False).to(device)

    print("🔧  FX replacing ...")
    hookable = make_inception_hookable(net)

    # 1) 统计替换后的模块数量
    n_relu = sum(1 for m in hookable.modules() if isinstance(m, ReLU))
    n_cat  = sum(1 for m in hookable.modules() if isinstance(m, Cat))
    n_add  = sum(1 for m in hookable.modules() if isinstance(m, Add))
    print(f"📊  After replace:  ReLU={n_relu}  Cat={n_cat}  Add={n_add}")

    # 2) 注册钩子，验证能否正常触发
    hook_log = []
    def fw_hook(m, inp, out):
        hook_log.append(m.__class__.__name__)

    for m in hookable.modules():
        if isinstance(m, (ReLU, Cat, Add)):
            m.register_forward_hook(fw_hook)

    x = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        _ = hookable(x)

    triggered = set(hook_log)
    print(f"📊  Hook triggered: {triggered}")

    # 3) 断言：至少抓到了 ReLU，才算成功
    assert n_relu > 10,  "FX replace ReLU failed!"
    assert n_cat >= 1,   "FX replace Cat failed!"
    assert "ReLU" in triggered, "ReLU hook not triggered!"
    print("✅  Unit test PASSED — FX replace & hook work well on NPU.")


if __name__ == "__main__":
    test_fx_replace()