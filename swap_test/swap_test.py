# test_swap_public_api.py
import sys, torch, torch.nn as nn
import torch_npu
from torch_npu.contrib import transfer_to_npu

# ----------- 1. NPU 环境 ----------
try:
    import acl
except ImportError:
    raise SystemExit("❌ 未安装 PyACL，请 source set_env.sh")
acl.rt.set_device(0)
acl.init()

# ----------- 2. 公开 API 路径：D→H → free → malloc → H→D → 换壳 ----------
def replace_npu_storage(tensor):
    size    = tensor.numel() * tensor.element_size()
    old_ptr = tensor.data_ptr()
    print(f"[REPLACE-0] 开始 | old_ptr={old_ptr:#x}")

    # 1. D→H：字节容器 → reinterpret → reshape → copy
    host_bytes  = torch.empty(size, dtype=torch.uint8, pin_memory=True)
    host_tensor = host_bytes.view(tensor.dtype).reshape(tensor.shape)   # ← 关键修复
    host_tensor.copy_(tensor)
    print(f"[REPLACE-1] D→H 完成 | host={host_tensor.data_ptr():#x}")

    # 2. free 旧 NPU
    acl.rt.free(old_ptr)
    print(f"[REPLACE-2] 已 free 旧 NPU | {old_ptr:#x}")

    # 3. PyTorch malloc 新 NPU
    new_tensor = torch.empty(tensor.shape, dtype=tensor.dtype, device='npu')
    new_ptr = new_tensor.data_ptr()
    print(f"[REPLACE-3] PyTorch malloc 新块 | new_ptr={new_ptr:#x}")

    # 4. H→D：同样 reinterpret/reshape
    new_tensor.copy_(host_tensor)        # ← 同样修复
    print(f"[REPLACE-4] H→D 完成 | 数据已拷回 {new_ptr:#x}")

    # 5. 公开 API 换壳
    tensor.set_(new_tensor.storage(), tensor.storage_offset(), tensor.size(), tensor.stride())
    print(f"[REPLACE-5] tensor 换壳完成 | tensor.data_ptr()={tensor.data_ptr():#x}")
    return new_ptr

# ----------- 3. 钩子：先跑原 D→H→D，再跑「换壳」 ----------
def make_hook(name):
    def post_hook(m, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        ptr  = t.data_ptr()
        size = t.numel() * t.element_size()
        mean_b = t.mean().item()

        # 3.2 公开 API 换壳
        replace_npu_storage(t)
        print(f"[PyACL-REPLACE] {name} 现在用新 ptr={t.data_ptr():#x} 训练线程自动可见")
    return post_hook

def register_hook(model):
    for name, m in model.named_modules():
        if type(m) not in {nn.Conv2d, nn.Linear}: continue
        m.register_forward_hook(make_hook(name))

# ----------- 4. 最小前向 ----------
if __name__ == "__main__":
    x = torch.randn(2, 3, 32, 32).npu()
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1)
    ).npu()

    register_hook(model)
    with torch.no_grad():
        out = model(x)
    print("=== 公开 API 换壳 + 日志完成 ===")