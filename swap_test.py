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
        self.events: List[Tuple[torch.Tensor, torch.Tensor]] = []  # (tensor, cpu_buf)

    def clear(self):
        self.events.clear()

GLOBAL_EVENT = GlobalVectorEvent()

class SwapTensor:
    def __init__(self, tensor, layer_name):
        self.tensor = tensor
        self.size = tensor.size()
        self.storage_size = tensor.storage().size()
        self.tensor_cpu = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True, device='cpu')

        self.d2h_event = None
        self.h2d_event = torch.npu.Event()

        self.stat = "device"
        self.layer_name = layer_name

        self.prefetch_data_ptr = tensor.data_ptr()
        self.storage_data_ptr = tensor.storage().data_ptr()
        self.layer_id = None
        self.first_tensor = False
        self.last_tensor = False
        self.is_slice_tensor = tensor.storage().size() != tensor.numel()
        self.stream = None
        self.layer_index = 0

    # device to host
    def launch_d2h(self, stream):
        if self.stat != "device":
            return
        forward_event = torch.npu.Event()
        forward_event.record()
        with torch.no_grad():
            with torch_npu.npu.stream(stream):
                stream.wait_event(forward_event)
                if self.is_slice_tensor:
                    self.tensor_cpu.copy_(self.tensor, non_blocking=True)
                else:
                    self.tensor_cpu.storage().copy_(self.tensor.storage(), non_blocking=True)
                self.stat = "d2h"

    # synchronize d2h and resize 0
    def wait_d2h_finished(self, stream, need_wait=False):
        if self.stat != "d2h":
            return
        if need_wait:
            torch.npu.current_stream().wait_stream(stream)
            torch.npu.default_stream().wait_stream(stream)
        self.tensor.storage().resize_(0)
        self.stat = "host"

    # resize storage_size and host to device
    def launch_h2d(self, stream, flag):
        if self.stat != "host":
            return
        backward_event = torch.npu.Event()
        backward_event.record()
        if flag:
            self.tensor.storage().resize_(self.storage_size)
        with torch.no_grad():
            with torch_npu.npu.stream(stream):
                stream.wait_event(backward_event)
                if self.is_slice_tensor:
                    self.tensor.copy_(self.tensor_cpu, non_blocking=True)
                else:
                    self.tensor.storage().copy_(self.tensor_cpu.storage(), non_blocking=True)
                self.h2d_event.record()
                self.stat = "h2d"

    # synchronize h2d
    def wait_h2d_finished(self, stream, need_wait=False):
        if self.stat != "h2d":
            return
        if need_wait:
            torch.npu.current_stream().wait_stream(stream)
            torch.npu.default_stream().wait_stream(stream)
        self.stat = "device"


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
                swap_tensor = SwapTensor(t, "Conv0")
                swap_tensor.launch_d2h(torch.npu.current_stream())
                swap_tensor.wait_d2h_finished(torch.npu.current_stream(), need_wait=True)
                print(f"[EVENT] Conv#{hook_count-1} | 卸载完成")
                # 记录到全局事件（句柄）
                GLOBAL_EVENT.events.append((t, swap_tensor))
        elif hook_count == 2:
            # 第二个卷积结束后从 CPU 搬回第一个卷积的输入 tensor
            if GLOBAL_EVENT.events:
                t, swap_tensor = GLOBAL_EVENT.events.pop(0)
                swap_tensor.launch_h2d(torch.npu.current_stream(), flag=True)
                swap_tensor.wait_h2d_finished(torch.npu.current_stream(), need_wait=True)
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
    x = torch.randn(2, 3, 8, 8).npu()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    with torch.enable_grad():
        out = net(x)
        loss = criterion(out, torch.randn_like(out))
        loss.backward()
        optimizer.step()

    print("---- 事件循环结束 ----")
    print(f"[FINAL] 事件列表长度: {len(GLOBAL_EVENT.events)}")