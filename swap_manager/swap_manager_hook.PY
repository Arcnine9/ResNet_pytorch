# swap_manager_hook.py
# MindSpeed 风格的 SwapManagerHook，但适配你的事件驱动

import torch
from .swap_manager import SwapManager


class SwapManagerHook:
    """
    MindSpeed 风格的 SwapManagerHook
    修改：移除自动 pack/unpack，改为配合你的 Module Hook
    """
    
    swap_hook = None

    def __init__(self):
        self.tensor_layer_name_prefix = ""
        self.pre_tensor_layer_name_prefix = ""
        self.swap_manager_modules = []
        
        # 你的扩展：当前层信息，用于 pack 时标记
        self._current_layer_name = None

    @staticmethod
    def unpack_hook(data):
        """标准 unpack，但你的方案中较少使用"""
        return SwapManager().unwrap_tensor(data)

    def pack_hook(self, origin_tensor):
        """
        修改后的 pack_hook：
        - 仍然调用 wrap_tensor（MindSpeed 标准）
        - 但移除 pre_tensor_is_allowed_swap 的自动层切换逻辑
        - 添加层名标记（用于你的查找）
        """
        swap_manager = SwapManager()
        
        # 标准 wrap
        wrapped = swap_manager.wrap_tensor(origin_tensor, pre_tensor_is_allowed_swap=False)
        
        # 你的扩展：如果知道当前层，标记到 SwappableTensor
        if hasattr(wrapped, 'set_tensor') and self._current_layer_name is not None:
            # 通过额外属性标记层名
            wrapped._pack_layer_name = self._current_layer_name
        
        return wrapped

    def hook_swap_manager_forward(self, forward_func, layer_name_prefix):
        """
        修改后的 hook：
        - 仍然使用 saved_tensors_hooks 上下文
        - 但层切换逻辑交给你控制
        """
        def custom_forward(*args, **kargs):
            # 设置当前层名（用于 pack_hook 标记）
            self._current_layer_name = layer_name_prefix
            
            with torch.autograd.graph.saved_tensors_hooks(self.pack_hook, self.unpack_hook):
                return forward_func(*args, **kargs)
        
        return custom_forward

    def reset_tensor_layer_info(self):
        self.tensor_layer_name_prefix = ""
        self.pre_tensor_layer_name_prefix = ""
        self._current_layer_name = None

    def reset_swap_manager_modules(self):
        for m in self.swap_manager_modules:
            m.forward = m.no_checkpoint_swap_forward
        self.swap_manager_modules.clear()


def get_swap_hook():
    if SwapManagerHook.swap_hook is None:
        SwapManagerHook.swap_hook = SwapManagerHook()
    return SwapManagerHook.swap_hook


# ==================== 你的注册函数（替换 MindSpeed 的 register_recursive_apply）====================

def register_swap_hooks(model, target_layers: set, swap_hook: SwapManagerHook = None):
    """
    你的注册函数：按层名匹配，而不是 ctx 继承
    
    target_layers: 需要 swap 的层名集合，来自你的 trace 文件
    """
    if swap_hook is None:
        swap_hook = get_swap_hook()
    
    for name, module in model.named_modules():
        if name in target_layers:
            # 保存原始 forward
            module.no_checkpoint_swap_forward = module.forward
            
            # 替换为带 saved_tensors_hooks 的版本
            module.forward = swap_hook.hook_swap_manager_forward(
                module.forward, 
                layer_name_prefix=name
            )
            
            swap_hook.swap_manager_modules.append(module)
            print(f"[register_swap_hooks] Registered: {name}")


def reset_swap_hooks(swap_hook: SwapManagerHook = None):
    """清理"""
    if swap_hook is None:
        swap_hook = get_swap_hook()
    swap_hook.reset_swap_manager_modules()
    swap_hook.reset_tensor_layer_info()