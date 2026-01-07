import torch
import torch.nn as nn
from typing import List, Tuple, Dict
import torch_npu
from torch_npu.contrib import transfer_to_npu
import acl

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

# ========== SwapManager 类 ==========
class SwapManager:
    def __init__(self):
        self.swap_tensors: Dict[int, SwapTensor] = {}  # key: tensor_id, value: SwapTensor
        self.issued_time = 0  # 当前已处理的事件时间戳

    def add_swap_tensor(self, tensor_id: int, tensor: torch.Tensor, layer_name: str):
        """添加一个新的 SwapTensor 对象"""
        if tensor_id not in self.swap_tensors:
            self.swap_tensors[tensor_id] = SwapTensor(tensor, layer_name)
        else:
            raise ValueError(f"Tensor ID {tensor_id} already exists in SwapManager.")

    def get_swap_tensor(self, tensor_id: int) -> SwapTensor:
        """根据 tensor_id 获取 SwapTensor 对象"""
        return self.swap_tensors.get(tensor_id, None)

    def launch_d2h(self, tensor_id: int, stream):
        """触发 device to host 操作"""
        swap_tensor = self.get_swap_tensor(tensor_id)
        if swap_tensor:
            swap_tensor.launch_d2h(stream)
            print(f"[EVENT] Issued Time: {self.issued_time} | Tensor: {tensor_id} | From: In_gpu, To: In_cpu")
            self.issued_time += 1

    def wait_d2h_finished(self, tensor_id: int, stream, need_wait=False):
        """等待 device to host 操作完成"""
        swap_tensor = self.get_swap_tensor(tensor_id)
        if swap_tensor:
            swap_tensor.wait_d2h_finished(stream, need_wait)

    def launch_h2d(self, tensor_id: int, stream, flag):
        """触发 host to device 操作"""
        swap_tensor = self.get_swap_tensor(tensor_id)
        if swap_tensor:
            swap_tensor.launch_h2d(stream, flag)
            print(f"[EVENT] Issued Time: {self.issued_time} | Tensor: {tensor_id} | From: In_cpu, To: In_gpu")
            self.issued_time += 1

    def wait_h2d_finished(self, tensor_id: int, stream, need_wait=False):
        """等待 host to device 操作完成"""
        swap_tensor = self.get_swap_tensor(tensor_id)
        if swap_tensor:
            swap_tensor.wait_h2d_finished(stream, need_wait)

    def clear(self):
        """清空所有 SwapTensor 对象"""
        self.swap_tensors.clear()
        self.issued_time = 0