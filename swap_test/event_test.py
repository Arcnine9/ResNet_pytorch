# swap_event_engine.py
import torch
import torch.nn as nn
from typing import List, Tuple
import torch_npu
from torch_npu.contrib import transfer_to_npu
import acl

# ========== 全局向量对象 ==========
class GlobalVectorEvent:
    def __init__(self):
        self.events: List[Tuple[torch.Tensor, torch.Tensor]] = []   # (tensor, cpu_buf)

    def clear(self):
        self.events.clear()

GLOBAL_EVENT = GlobalVectorEvent()

# ========== D→H 卸载函数 ==========
def d2h_unload(tensor: torch.Tensor) -> torch.Tensor:
    """D→H：把 tensor 搬出 NPU，释放 ptr"""
    size = tensor.numel() * tensor.element_size()
    # 1. 申请同形 CPU 缓冲（句柄）
    cpu_buf = torch.empty_like(tensor, device='cpu')
    # 2. D→H 拷贝（原地换 storage）
    cpu_buf.copy_(tensor)                       # D→H
    # 3. 真正释放旧 NPU 内存
    acl.rt.free(tensor.data_ptr())
    print(f"[D2H] 卸载完成 | tensor.ptr={tensor.data_ptr():#x} | cpu_buf.ptr={cpu_buf.data_ptr():#x}")
    return cpu_buf

# ========== H→D 预取函数 ==========
def h2d_prefetch(tensor: torch.Tensor, cpu_buf: torch.Tensor) -> None:
    """H→D：重新申请 NPU 内存，把 cpu_buf 搬回"""
    size = tensor.numel() * tensor.element_size()
    # 1. 重新申请 NPU 内存（可能拿到同 ptr，也可能不同）
    new_tensor = torch.empty_like(tensor, device='npu')
    new_ptr = new_tensor.data_ptr()
    print(f"[H2D] 重新 malloc | new_ptr={new_ptr:#x}")
    # 2. H→D 拷贝（原地换 storage）
    new_tensor.copy_(cpu_buf)                       # H→D
    print(f"[H2D] 预取完成 | tensor.ptr={new_tensor.data_ptr():#x} | cpu_buf.ptr={cpu_buf.data_ptr():#x}")
    # 3. 公开 API 换壳
    tensor.set_(new_tensor.storage(), tensor.storage_offset(), tensor.size(), tensor.stride())

# ========== Hook：根据计数器触发 D→H 或 H→D ==========
hook_count = 0  # 全局钩子计数器

def make_swap_event_hook():
    """Hook 根据计数器触发事件：先卸载，再预取"""
    def post_hook(m, inp, out):
        global hook_count
        hook_count += 1

        if hook_count == 1:
            # 第一个卷积结束后卸载输入 tensor
            t = inp[0]  # 第一个卷积的输入 tensor
            if isinstance(t, torch.Tensor) and t.is_cuda:
                cpu_buf = d2h_unload(t)
                print(f"[EVENT] Conv#{hook_count-1} | 卸载完成")
                # 记录到全局事件（句柄）
                GLOBAL_EVENT.events.append((t, cpu_buf))
        elif hook_count == 2:
            # 第二个卷积结束后从 CPU 搬回第一个卷积的输入 tensor
            if GLOBAL_EVENT.events:
                t, cpu_buf = GLOBAL_EVENT.events.pop(0)
                h2d_prefetch(t, cpu_buf)
                print(f"[EVENT] Conv#{hook_count-1} | 预取完成")

    return post_hook

# ========== 使用示例 ==========
if __name__ == "__main__":
    # 1. 模型（两个卷积）
    net = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.Conv2d(16, 32, 3, padding=1)
    ).npu()

    # 2. 注册钩子
    net[0].register_forward_hook(make_swap_event_hook())  # 为第一个卷积注册 hook
    net[1].register_forward_hook(make_swap_event_hook())  # 为第二个卷积注册 hook

    # 3. 训练循环（事件 = 函数调用）
    with torch.no_grad():
        out = net(torch.randn(2, 3, 8, 8).npu())
        print("---- 事件循环结束 ----")
        print(f"[FINAL] 事件列表长度: {len(GLOBAL_EVENT.events)}")