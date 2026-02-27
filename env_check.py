import os
import sys
import traceback
import torch

def banner(msg):
    print("\n" + "=" * 80)
    print(msg)
    print("=" * 80)

def main():
    banner("Ascend NPU Environment Sanity Check")

    # 1. Python & Torch 基本信息
    print("[Basic Info]")
    print("Python:", sys.version.replace("\n", " "))
    print("Torch :", torch.__version__)

    # 2. NPU 可见性
    banner("Step 1: NPU Visibility Check")
    try:
        import torch_npu
        print("torch_npu version:", torch_npu.__version__)
        available = torch.npu.is_available()
        print("torch.npu.is_available():", available)
        if not available:
            print("[FAIL] NPU not available at runtime")
            return
    except Exception as e:
        print("[FAIL] torch_npu import failed")
        traceback.print_exc()
        return

    # 3. Ascend 环境变量检查
    banner("Step 2: Ascend Environment Variables")
    for k in ["ASCEND_HOME", "ASCEND_OPP_PATH", "LD_LIBRARY_PATH"]:
        print(f"{k} =", os.environ.get(k, "<NOT SET>"))

    # 4. 最小 Conv2D 测试（FP32，最稳态路径）
    banner("Step 3: Minimal Conv2D Test on NPU (FP32)")

    try:
        device = "npu:0"

        # 显式 FP32
        x = torch.randn(
            2, 3, 224, 224,
            dtype=torch.float32,
            device=device
        )

        conv = torch.nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=True,
        ).to(device, dtype=torch.float32)

        # 强制同步，避免 lazy error
        y = conv(x)
        torch.npu.synchronize()

        print("[PASS] FP32 Conv2D executed successfully on NPU")
        print("Output shape:", tuple(y.shape))
        print("Output dtype:", y.dtype)

    except Exception:
        print("[FAIL] FP32 Conv2D execution failed on NPU")
        traceback.print_exc()

        print("\n[Conclusion]")
        print(
            "FP32 Conv2D failed on Ascend NPU.\n"
            "This definitively rules out FP16 / AMP issues.\n"
            "The Ascend ACL / OPP / operator registry is NOT functional "
            "on this node.\n"
            "This is a system-level environment problem, not a PyTorch issue."
        )

        return

    banner("Final Result")
    print("[SUCCESS] Ascend NPU environment is FUNCTIONAL")

if __name__ == "__main__":
    main()
