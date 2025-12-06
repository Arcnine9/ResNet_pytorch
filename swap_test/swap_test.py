# test_pyacl_hook.py
import sys
import torch
import torch.nn as nn
from swap_manager.hook import register_all_hooks   # 你的原 hooks.py

# ----------- 1. 环境自检 -----------
try:
    import acl
except ImportError:
    raise SystemExit("❌ 未安装 PyACL，请 source /usr/local/Ascend/ascend-toolkit/set_env.sh")

if acl.rt.get_device_count() == 0:
    raise SystemExit("❌ 未发现 Ascend 设备，请检查驱动")

acl.rt.set_device(0)          # 绑定设备 0
acl.init()                    # 一次性初始化

# ----------- 2. 纯 Python 同步 D→H→D + 计时 -----------
def pyacl_copy_dhd(dev_ptr: int, size: int) -> float:
    """同步 D→H→D，返回耗时（μs）"""
    # 主机页锁定
    host_ptr, ret = acl.rt.malloc_host(size)
    assert ret == 0, "malloc_host failed"

    # 事件计时
    start, ret = acl.rt.create_event()
    end, ret = acl.rt.create_event()
    acl.rt.record_event(start, 0)          # stream=0

    # D→H
    acl.rt.memcpy(host_ptr, size, dev_ptr, size, acl.rt.MEMCPY_DEVICE_TO_HOST)
    # H→D
    acl.rt.memcpy(dev_ptr, size, host_ptr, size, acl.rt.MEMCPY_HOST_TO_DEVICE)

    acl.rt.record_event(end, 0)
    acl.rt.synchronize_event(end)
    elapsed_ms = acl.rt.event_elapsed_time(start, end)
    acl.rt.free_host(host_ptr)
    acl.rt.destroy_event(start)
    acl.rt.destroy_event(end)
    return elapsed_ms * 1000.0   # us

# ----------- 3. 与训练 hook 一致的 post-hook -----------
def make_pyacl_test_hook(name):
    def post_hook(m, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        ptr  = t.data_ptr()
        size = t.numel() * t.element_size()
        mean_before = t.mean().item()

        # 4. 实时调用 PyACL 同步搬运
        elapsed_us = pyacl_copy_dhd(ptr, size)

        mean_after = t.mean().item()

        # 5. 打印结果
        print(f"[PyACL-TEST] {name} | ptr={ptr:#x} | size={size}B | "
              f"mean_before={mean_before:.6f} | mean_after={mean_after:.6f} | "
              f"copy_time={elapsed_us:.3f}us")
    return post_hook

def register_pyacl_test_hook(model):
    for name, module in model.named_modules():
        if type(module) not in {nn.Conv2d, nn.Linear}:
            continue
        module.register_forward_hook(make_pyacl_test_hook(name))

# ----------- 6. 最小前向（不训练） -----------
if __name__ == "__main__":
    # 让 python 找到 hooks.py
    sys.path.insert(0, '.')

    # 最小模型 + 数据
    x = torch.randn(2, 3, 32, 32).npu()
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1)
    ).npu()

    register_pyacl_test_hook(model)   # 只挂测试钩子
    with torch.no_grad():
        out = model(x)                # 一次前向即可
    print("=== PyACL 搬运测试完成 ===")