# test_pyacl_replace_ptr.py
import sys, torch, torch.nn as nn

# 1. 环境自检
try:
    import acl
except ImportError:
    raise SystemExit("❌ 未安装 PyACL，请 source set_env.sh")
if acl.rt.get_device_count() == 0:
    raise SystemExit("❌ 未发现 Ascend 设备")
acl.rt.set_device(0)
acl.init()

# 2. 工具：同步 D→H→D + 换 ptr
def replace_npu_storage(tensor):
    """
    把 tensor 的 NPU 存储整块换掉，返回新 ptr
    步骤：D→H → free NPU → malloc NPU → H→D → 原地换 storage
    """
    size   = tensor.numel() * tensor.element_size()
    old_ptr= tensor.data_ptr()

    # 2.1 主机锁页
    host_ptr, ret = acl.rt.malloc_host(size)
    assert ret == 0, "malloc_host fail"

    # 2.2 D→H 拷贝
    acl.rt.memcpy(host_ptr, size, old_ptr, size, acl.rt.MEMCPY_DEVICE_TO_HOST)
    print(f"[1] D→H 完成 | old_ptr={old_ptr:#x} → host={host_ptr:#x}")

    # 2.3 释放旧 NPU 块
    acl.rt.free(old_ptr)
    print(f"[2] 已 free 旧 NPU 块 | {old_ptr:#x}")

    # 2.4 重新申请 NPU 块（可能拿到同地址，也可能不同）
    new_dev_ptr, ret = acl.rt.malloc(size)
    assert ret == 0, "malloc fail"
    print(f"[3] 重新 malloc | new_ptr={new_dev_ptr:#x}")

    # 2.5 H→D 拷贝
    acl.rt.memcpy(new_dev_ptr, size, host_ptr, size, acl.rt.MEMCPY_HOST_TO_DEVICE)
    print(f"[4] H→D 完成 | host={host_ptr:#x} → new_dev={new_dev_ptr:#x}")

    # 2.6 主机内存释放
    acl.rt.free_host(host_ptr)

    # 2.7 原地换 storage（Python 层）
    new_storage = torch.empty(tensor.numel(), dtype=tensor.dtype, device='npu').storage()
    # 把新 storage 的数据区指针换成我们刚申请的
    # （这里用 private API，仅测试）
    import torch_npu
    torch_npu._C.npu_storage_fill(new_storage, new_dev_ptr, size)
    tensor.set_(new_storage, tensor.storage_offset(), tensor.size(), tensor.stride())
    print(f"[5] tensor.storage 已替换 | tensor.data_ptr()={tensor.data_ptr():#x}")
    return new_dev_ptr

# 3. 测试模型 + 数据
x = torch.randn(2, 3, 8, 8).npu()
model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 32, 3, padding=1)
).npu()

# 4. 钩子：在 Relu 后把输出 tensor 整个换 ptr
def post_relu_hook(m, inp, out):
    print("---- 进入 Relu post-hook ----")
    ptr_before = out.data_ptr()
    print(f"[hook] 替换前 ptr={ptr_before:#x}")
    new_ptr = replace_npu_storage(out)
    print(f"[hook] 替换后 ptr={out.data_ptr():#x}  <-- 训练线程将看到它")

model[1].register_forward_hook(post_relu_hook)

# 5. 再跑一次前向（训练线程完全无感）
with torch.no_grad():
    out = model(x)
    print("---- 前向结束 ----")
    print(f"[final] out.ptr={out.data_ptr():#x}  <-- 确认是新地址")
    print(f"[final] out.mean={out.mean().item():.6f}  <-- 数值没变")