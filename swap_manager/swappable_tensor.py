# swappable_tensor.py
# 完全复用 MindSpeed 的 SwappableTensor，不做任何修改

import torch
import torch_npu


class SwappableTensor(torch.Tensor):
    """
    MindSpeed 标准 SwappableTensor
    完全复用，只添加 tensor_id 用于调试
    """
    
    @classmethod
    def __new__(cls, tensor, *args, **kwargs):
        data = torch.Tensor([id(tensor)])
        return torch.Tensor._make_subclass(cls, data, False)

    def __init__(self, tensor, tensor_id=None):
        self.tensor_id = tensor_id  # 仅用于调试，不影响功能
        self.id_key = None
        self.inner_tensor = None
        self.inner_tensor_bro_keys = []
        self.inner_tensor_cpu_data = None
        self.inner_tensor_data_ptr = None
        self.inner_tensor_origin_storage_size = 0
        self.is_allowed_swap = False
        self._device = None
        self._location = None

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return super().__torch_function__(func, types, args, kwargs)

    def set_tensor(self, id_key, tensor):
        """MindSpeed 标准接口"""
        self.id_key = id_key
        self.inner_tensor = tensor
        self.inner_tensor_data_ptr = tensor.data_ptr()
        self.inner_tensor_origin_storage_size = tensor.storage().size()
        self._location = "device"
        self._device = tensor.device

    def get_tensor(self):
        """MindSpeed 标准接口"""
        return self.inner_tensor

    def set_tensor_location(self, location):
        """MindSpeed 标准接口"""
        self._location = location

    def get_location(self):
        """MindSpeed 标准接口"""
        return self._location

    def trans_to_cpu(self):
        """
        MindSpeed 标准接口：同步搬运到 CPU
        包含 resize_(0) 释放 device storage
        """
        with torch.no_grad():
            self.inner_tensor_cpu_data = self.inner_tensor.cpu()
            self.inner_tensor.storage().resize_(0)
            self._location = "cpu"

    def trans_to_device(self):
        """
        MindSpeed 标准接口：同步搬运回 device
        包含 resize_(origin_size) 恢复 storage
        """
        with torch.no_grad():
            self.inner_tensor.storage().resize_(self.inner_tensor_origin_storage_size)
            self.inner_tensor.copy_(self.inner_tensor_cpu_data)
            self._location = "device"