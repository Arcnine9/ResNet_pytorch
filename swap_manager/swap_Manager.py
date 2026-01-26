# swap_manager.py
# 简化版：移除分片 tensor 处理，保留核心安全检查和 MindSpeed 接口

import os
import time
from typing import Dict, Optional
import torch
import torch_npu

from .swappable_tensor import SwappableTensor


def get_tensor_mem_size(tensor):
    return tensor.numel() * tensor.element_size()


class SwapManagerMeta(type):
    """单例模式"""
    swap_manager_instance = {}

    def __call__(cls, *args, **kwargs):
        device = kwargs.get('device', None)
        key = device if device is not None else 'default'
        if key not in cls.swap_manager_instance:
            instance = super().__call__(*args, **kwargs)
            cls.swap_manager_instance[key] = instance
        return cls.swap_manager_instance[key]


class SwapManager(metaclass=SwapManagerMeta):
    def __init__(self, device=None):
        # 简化：只保留两个存储区
        self.device_tensors: Dict[int, SwappableTensor] = {}
        self.host_tensors: Dict[int, SwappableTensor] = {}
        
        self.device = device if device is not None else torch_npu.npu.current_device()
        self.total_swap_out_size = 0
        
        print(f"[SwapManager] device={self.device}")

    # ========================= 安全检查（保留）=========================
    
    @staticmethod
    def is_allowed_wrap_tensor(tensor):
        """
        保留 MindSpeed 的安全检查：
        - 拒绝已经是 SwappableTensor（双重包装）
        - 拒绝非连续内存
        - 拒绝过小 tensor（<1024B）
        - 拒绝切片 tensor（关键！防止 resize 出错）
        - 拒绝叶子节点（参数）
        """
        if isinstance(tensor, SwappableTensor):
            return False
        if not tensor.is_contiguous():
            return False
        
        min_size = int(os.getenv('MIN_SWAP_TENSOR_SIZE', 1024))
        if get_tensor_mem_size(tensor) < min_size:
            return False
        
        # 关键：拒绝切片 tensor（storage 不连续）
        if tensor.storage_offset() != 0 or tensor.storage().size() != tensor.numel():
            print("[DEBUG] Rejecting sliced tensor for swap")
            return False
        
        # 拒绝参数（叶子节点）
        if tensor.grad_fn is None:
            return False
        
        return True

    # ========================= 核心接口 =========================
    
    def wrap_tensor(self, tensor_id: int, tensor: torch.Tensor) -> torch.Tensor:
        """
        包装 tensor（Extract 事件）
        
        简化：移除 pre_tensor_is_allowed_swap 参数，
        因为 trace 已经精确决定了每个 tensor 的 swap 时机
        """
        if not self.is_allowed_wrap_tensor(tensor):
            return tensor
        
        # 检查是否已存在
        if tensor_id in self.device_tensors or tensor_id in self.host_tensors:
            existing = self.device_tensors.get(tensor_id) or self.host_tensors.get(tensor_id)
            return existing
        
        # 创建 SwappableTensor
        wrapped = SwappableTensor(tensor, tensor_id=tensor_id)
        wrapped.set_tensor(tensor_id, tensor)  # 用 tensor_id 作为 key
        
        self.device_tensors[tensor_id] = wrapped
        
        print(f"[wrap {tensor_id}] shape={tensor.shape}, storage_size={wrapped.inner_tensor_origin_storage_size}")
        return wrapped

    def unwrap_tensor(self, tensor) -> torch.Tensor:
        """
        解包 tensor（获取实际 tensor）
        
        简化：如果 tensor 在 host，自动触发 H2D（同步）
        """
        if not isinstance(tensor, SwappableTensor):
            return tensor
        
        tid = tensor.id_key
        
        # 在 device：直接返回
        if tid in self.device_tensors:
            return tensor.get_tensor()
        
        # 在 host：触发 H2D（应急情况，正常应由事件触发）
        if tid in self.host_tensors:
            print(f"[WARNING] unwrap_tensor {tid}: in host, emergency H2D")
            self.host_tensors.pop(tid)
            tensor.trans_to_device()
            self.device_tensors[tid] = tensor
            return tensor.get_tensor()
        
        raise RuntimeError(f"Tensor {tid} not found")

    # ========================= 事件驱动接口（简化）=========================
    
    def execute_d2h(self, tensor_id: int):
        """
        执行 D2H（"In_cpu" -> "In_cpu" 事件）
        
        简化：直接调用 trans_to_cpu，不处理分片 tensor
        """
        if tensor_id not in self.device_tensors:
            print(f"[ERROR] execute_d2h: {tensor_id} not found")
            return False
        
        wrapped = self.device_tensors[tensor_id]
        
        # 安全检查
        if wrapped.get_location() == "cpu":
            print(f"[WARNING] execute_d2h: {tensor_id} already on cpu")
            return True
        
        # 执行 D2H（同步）
        wrapped.trans_to_cpu()
        
        # 移动存储区
        self.device_tensors.pop(tensor_id)
        self.host_tensors[tensor_id] = wrapped
        
        # 统计
        size = wrapped.inner_tensor_origin_storage_size * wrapped.inner_tensor.element_size()
        self.total_swap_out_size += size
        
        print(f"[execute_d2h {tensor_id}] size={size} bytes")
        return True

    def execute_h2d(self, tensor_id: int):
        """
        执行 H2D（"In_cpu" -> "In_gpu" 事件）
        
        简化：直接调用 trans_to_device
        """
        if tensor_id not in self.host_tensors:
            # 可能已经在 device（重复事件）
            if tensor_id in self.device_tensors:
                print(f"[execute_h2d {tensor_id}] already on device")
                return True
            print(f"[ERROR] execute_h2d: {tensor_id} not found")
            return False
        
        wrapped = self.host_tensors[tensor_id]
        
        # 执行 H2D（同步）
        wrapped.trans_to_device()
        
        # 移动存储区
        self.host_tensors.pop(tensor_id)
        self.device_tensors[tensor_id] = wrapped
        
        print(f"[execute_h2d {tensor_id}] completed")
        return True

    def check_on_device(self, tensor_id: int):
        """
        检查 tensor 是否在 device（"In_gpu" -> "In_gpu" 事件）
        
        简化：只做检查，不处理分片
        """
        if tensor_id in self.device_tensors:
            location = self.device_tensors[tensor_id].get_location()
            if location == "device":
                print(f"[check {tensor_id}] confirmed on device")
                return True
            else:
                print(f"[WARNING] check {tensor_id}: location={location}")
                return False
        
        if tensor_id in self.host_tensors:
            print(f"[ERROR] check {tensor_id}: found in host!")
            return False
        
        print(f"[ERROR] check {tensor_id}: not found")
        return False

    # ========================= 清理 =========================
    
    def clear(self):
        """Batch 结束清理"""
        for wrapped in list(self.device_tensors.values()):
            wrapped.inner_tensor_cpu_data = None
            wrapped.inner_tensor = None
        for wrapped in list(self.host_tensors.values()):
            wrapped.inner_tensor_cpu_data = None
            wrapped.inner_tensor = None
        
        self.device_tensors.clear()
        self.host_tensors.clear()
        self.total_swap_out_size = 0
        print("[clear] completed")
    
    # 兼容接口
    reset_swap_manager_tensors = clear