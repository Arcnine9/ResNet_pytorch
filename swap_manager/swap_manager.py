# swap_manager.py
# MindSpeed 完整架构 + 你的最小扩展（tid 标记）

import os
import time
from copy import deepcopy
from typing import Dict, Optional
import torch
import torch_npu

from .swappable_tensor import SwappableTensor


def hum_convert(value):
    """MindSpeed 标准"""
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    origin_value = value
    for unit in units:
        if (value / 1024.0) < 1:
            return "%.2f%s" % (value, unit)
        value = value / 1024.0
    return "%.2f%s" % (origin_value, units[0])


def get_tensor_mem_size(tensor):
    """MindSpeed 标准"""
    return tensor.numel() * tensor.element_size()


class SwapManagerMeta(type):
    """MindSpeed 标准：单例模式"""
    swap_manager_instance = {}

    def __call__(cls, *args, **kwargs):
        device = kwargs.get('device', None)
        key = device if device is not None else 'default'
        
        if key not in cls.swap_manager_instance:
            instance = super().__call__(*args, **kwargs)
            cls.swap_manager_instance[key] = instance
        return cls.swap_manager_instance[key]


class SwapManager(metaclass=SwapManagerMeta):
    """
    MindSpeed 标准 SwapManager
    最小扩展：添加 tid 标记支持
    """
    
    def __init__(self, device=None):
        # MindSpeed 标准属性
        self.host_tensors: Dict[int, SwappableTensor] = {}
        self.device_tensors: Dict[int, SwappableTensor] = {}
        self.total_swap_out_size = 0
        self.origin_layers_peak_memory = {}
        self.policy_peak_memory = {}
        self.layers_interval_peak_memory = {}
        self.cur_pre_hook_layer_name = ""
        self.cur_post_hook_layer_name = ""
        self.swap_status = False
        
        # 你的扩展：设备绑定
        self.device = device if device is not None else torch_npu.npu.current_device()
        
        # 你的扩展：tid 反向查找（id(tensor) -> key）
        self._tensor_id_map: Dict[int, int] = {}

    # ==================== MindSpeed 标准接口 ====================
    
    @staticmethod
    def is_allowed_wrap_tensor(tensor):
        """MindSpeed 标准"""
        if isinstance(tensor, SwappableTensor):
            return False
        if not tensor.is_contiguous():
            return False
        
        config = os.getenv('MIN_SWAP_TENSOR_SIZE')
        min_swap_tensor_size = 1024
        if config is not None:
            min_swap_tensor_size = max(min_swap_tensor_size, int(config))
        if get_tensor_mem_size(tensor) < min_swap_tensor_size:
            return False
        
        if tensor.storage_offset() != 0 or tensor.storage().size() != tensor.numel():
            return False
        
        if tensor.grad_fn is None:
            return False
        
        return True

    def change_manager_tensor_status_to_allowed_swap(self):
        """MindSpeed 标准"""
        for k in self.device_tensors.keys():
            self.device_tensors[k].is_allowed_swap = True

    def wrap_tensor(self, tensor, pre_tensor_is_allowed_swap=False):
        """
        MindSpeed 标准 wrap_tensor
        最小扩展：记录 id(tensor) -> key 映射
        """
        if pre_tensor_is_allowed_swap:
            self.change_manager_tensor_status_to_allowed_swap()
        
        if not self.is_allowed_wrap_tensor(tensor):
            return tensor
        
        wrapped_tensor = SwappableTensor(tensor)
        key = time.time()
        wrapped_tensor.set_tensor(key, tensor)
        
        # 你的扩展：记录映射，用于后续查找
        self._tensor_id_map[id(tensor)] = key
        
        self.device_tensors[key] = wrapped_tensor
        
        return wrapped_tensor

    def unwrap_tensor(self, tensor):
        """MindSpeed 标准"""
        if not isinstance(tensor, SwappableTensor):
            return tensor
        
        key = tensor.id_key
        
        if key in self.host_tensors.keys():
            self.host_tensors.pop(key)
            if tensor.get_tensor().storage().size() == 0:
                self.move_shard_tensor_to_device(tensor)
        else:
            self.device_tensors.pop(key, None)
        
        return tensor.get_tensor()

    def move_shard_tensor_to_host(self, bro_key, bro_tensor):
        """MindSpeed 标准"""
        move_count = 0
        device_tensors_keys = list(self.device_tensors.keys())
        
        for key in device_tensors_keys:
            tensor = self.device_tensors[key]
            if tensor.inner_tensor_data_ptr == bro_tensor.inner_tensor_data_ptr:
                self.device_tensors.pop(key)
                tensor.set_tensor_location("cpu")
                tensor.inner_tensor_bro_keys.append(bro_key)
                bro_tensor.inner_tensor_bro_keys.append(key)
                self.host_tensors[key] = tensor
                move_count += 1
        
        self.host_tensors[bro_key] = bro_tensor
        return move_count

    def move_shard_tensor_to_device(self, tensor):
        """MindSpeed 标准"""
        cap_tensor = tensor
        
        if tensor.inner_tensor_cpu_data is None:
            cap_key = tensor.inner_tensor_bro_keys[0]
            try:
                cap_tensor = self.host_tensors[cap_key]
            except KeyError:
                print(f"[ERROR] The key doesn't exist.")
                raise
        
        cap_tensor.trans_to_device()
        
        if cap_tensor.id_key != tensor.id_key:
            cap_tensor.inner_tensor_bro_keys.remove(tensor.id_key)
            self.host_tensors.pop(cap_tensor.id_key)
            self.device_tensors[cap_tensor.id_key] = cap_tensor
        
        for key in cap_tensor.inner_tensor_bro_keys:
            bro_tensor = self.host_tensors.pop(key)
            bro_tensor.set_tensor_location("device")
            self.device_tensors[key] = bro_tensor

    def reset_swap_manager_tensors(self):
        """MindSpeed 标准"""
        for st in self.device_tensors.values():
            st.inner_tensor_cpu_data = None
            st.inner_tensor = None
        for st in self.host_tensors.values():
            st.inner_tensor_cpu_data = None
            st.inner_tensor = None
        
        self.device_tensors.clear()
        self.host_tensors.clear()
        self._tensor_id_map.clear()
        self.total_swap_out_size = 0
        self.swap_status = False
        self.cur_pre_hook_layer_name = ""
        self.cur_post_hook_layer_name = ""

    # ==================== 你的扩展接口 ====================
    
    def get_swappable_by_tensor(self, tensor: torch.Tensor) -> Optional[SwappableTensor]:
        """
        你的扩展：通过原始 tensor 查找 SwappableTensor
        
        使用预存的 id(tensor) -> key 映射，O(1)
        """
        # 方法 1：直接查映射
        key = self._tensor_id_map.get(id(tensor))
        if key is not None:
            # 可能在 device 或 host
            if key in self.device_tensors:
                return self.device_tensors[key]
            if key in self.host_tensors:
                return self.host_tensors[key]
        
        # 方法 2：兜底遍历（pack 时未记录的情况）
        for st in self.device_tensors.values():
            if st.inner_tensor is tensor:
                return st
        
        return None
    
    def mark_tid(self, tensor: torch.Tensor, tid: int) -> bool:
        """
        你的扩展：为 tensor 标记 tid
        """
        st = self.get_swappable_by_tensor(tensor)
        if st is not None:
            st.tensor_id = tid
            return True
        return False
    
    def get_by_tid(self, tid: int) -> Optional[SwappableTensor]:
        """
        你的扩展：通过 tid 查找（需要提前建立 tid -> tensor 映射）
        """
        for st in list(self.device_tensors.values()) + list(self.host_tensors.values()):
            if st.tensor_id == tid:
                return st
        return None