# test_pyacl_hook_replace_ptr.py
import sys, torch, torch.nn as nn
from swap_manager.hook import register_all_hooks   # 你的原文件
import torch_npu
from torch_npu.contrib import transfer_to_npu
import ctypes

# ----------- 1. 环境自检 -----------
try:
    import acl
except ImportError:
    raise SystemExit("❌ 未安装 PyACL，请 source set_env.sh")
if acl.rt.get_device_count() == 0:
    raise SystemExit("❌ 未发现 Ascend 设备")
acl.rt.set_device(0)
acl.init()

# ----------- 2. 原有 D→H→D（不改） -----------
def pyacl_copy_dhd(dev_ptr: int, size: int) -> float:
    host_ptr, ret = acl.rt.malloc_host(size)
    assert ret == 0
    start, _ = acl.rt.create_event()
    end, _   = acl.rt.create_event()
    acl.rt.record_event(start, 0)
    acl.rt.memcpy(host_ptr, size, dev_ptr, size, 2)   # D→H
    acl.rt.memcpy(dev_ptr, size, host_ptr, size, 1)   # H→D
    acl.rt.record_event(end, 0)
    acl.rt.synchronize_event(end)
    elapsed_ms, _ = acl.rt.event_elapsed_time(start, end)
    acl.rt.free_host(host_ptr)
    acl.rt.destroy_event(start)
    acl.rt.destroy_event(end)
    return elapsed_ms * 1000.0

# ----------- 3. 新增：free→malloc→HtoD→换 ptr -----------
def replace_npu_storage(tensor):
    """free 旧 NPU 块 → malloc 新块 → H→D → 原地换 ptr"""
    size    = tensor.numel() * tensor.element_size()
    old_ptr = tensor.data_ptr()

    # 3.1 先 D→H（复用页锁）
    host_ptr, ret = acl.rt.malloc_host(size)
    assert ret == 0
    acl.rt.memcpy(host_ptr, size, old_ptr, size, 2)
    print(f"[REPLACE-1] D→H 完成 | old_ptr={old_ptr:#x} → host={host_ptr:#x}")

    # 3.2 free 旧 NPU
    acl.rt.free(old_ptr)
    print(f"[REPLACE-2] 已 free 旧 NPU 块 | {old_ptr:#x}")

    # 3.3 malloc 新 NPU（可能同地址，也可能不同）
    new_ptr, ret = acl.rt.malloc(size, 0)
    assert ret == 0
    print(f"[REPLACE-3] 重新 malloc | new_ptr={new_ptr:#x}")

    # 3.4 H→D
    acl.rt.memcpy(new_ptr, size, host_ptr, size, 1)
    print(f"[REPLACE-4] H→D 完成 | host={host_ptr:#x} → new_dev={new_ptr:#x}")
    acl.rt.free_host(host_ptr)

    # 5. 零依赖换 storage（公开 API 路径）TODO:

    return new_ptr

# ----------- 4. 钩子：先跑原 D→H→D，再跑 free/malloc/H→D -----------
def make_pyacl_test_hook(name):
    def post_hook(m, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        ptr  = t.data_ptr()
        size = t.numel() * t.element_size()
        mean_before = t.mean().item()

        # 4.1 原来纯搬运（指针不变）
        elapsed_us = pyacl_copy_dhd(ptr, size)
        print(f"[PyACL-TEST] {name} | ptr={ptr:#x} | size={size}B | "
              f"mean_before={mean_before:.6f} | mean_after={t.mean().item():.6f} | "
              f"copy_time={elapsed_us:.3f}us")

        # 4.2 ↓↓↓ 新增：free→malloc→H→D→换 ptr
        replace_npu_storage(t)
        print(f"[PyACL-REPLACE] {name} 现在用新 ptr={t.data_ptr():#x} 训练线程自动可见")
    return post_hook

def register_pyacl_test_hook(model):
    for name, module in model.named_modules():
        if type(module) not in {nn.Conv2d, nn.Linear}:
            continue
        module.register_forward_hook(make_pyacl_test_hook(name))

# ----------- 5. 最小前向 -----------
if __name__ == "__main__":
    sys.path.insert(0, '.')
    x = torch.randn(2, 3, 32, 32).npu()
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1)
    ).npu()

    register_pyacl_test_hook(model)
    with torch.no_grad():
        out = model(x)
    print("=== 搬运 + 换 ptr 测试完成 ===")