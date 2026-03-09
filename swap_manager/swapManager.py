import torch
import torch_npu
from typing import Dict


# =========================
# EventPool：Event 复用池（强制 Device 绑定）
# =========================
class EventPool:
    """
    强制绑定到指定 device 的 Event 池
    """
    def __init__(self, num_events=32, device=None):
        if device is None:
            device = torch_npu.npu.current_device()
        self.device = device
        self.pool = []
        
        # 必须在指定 device 上下文中创建 Event
        with torch.npu.device(device):
            for _ in range(num_events):
                self.pool.append(torch.npu.Event())
        self.idx = 0

    def acquire(self):
        # 确保返回的 Event 在正确的 device 上下文中使用
        with torch.npu.device(self.device):
            evt = self.pool[self.idx]
            self.idx = (self.idx + 1) % len(self.pool)
            return evt

    def release(self, evt):
        # no-op：复用池不做生命周期管理
        pass


# =========================
# SwapTensor：纯状态对象
# =========================
class SwapTensor:
    """
    只保存状态，不保存 stream, 不做调度
    """
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor
        self.size = tensor.size()
        self.device = tensor.device  # 记录 tensor 的 device

        # 注意：TypedStorage 已 deprecated，但这里先保持最小改动
        self.storage_size = tensor.storage().size()

        self.is_slice_tensor = tensor.storage().size() != tensor.numel()

        if self.is_slice_tensor:
            self.tensor_cpu = torch.empty(
                tensor.shape,
                dtype=tensor.dtype,
                device="cpu"
            )
        else:
            self.tensor_cpu = torch.empty(
                self.storage_size,
                dtype=tensor.dtype,
                device="cpu"
            ).view(tensor.shape)

        # 状态机：device | d2h_inflight | host | h2d_inflight
        self.stat = "device"

        # event 只是句柄
        self.d2h_event = None
        self.h2d_event = None


# =========================
# SwapManager：唯一调度者（强制 Device 绑定）
# =========================
class SwapManager:
    def __init__(self, device=None, num_events: int = 32):
        """
        Args:
            device: 指定绑定的 device，如果为 None 则使用当前 device
            num_events: Event 池大小
        """
        if device is None:
            self.device = torch_npu.npu.current_device()
        else:
            self.device = device if isinstance(device, int) else device.index
        
        self.swap_tensors: Dict[int, SwapTensor] = {}
        
        # 在指定 device 上创建 EventPool
        with torch.npu.device(self.device):
            self.event_pool = EventPool(num_events=num_events, device=self.device)

    def _ensure_device_context(self):
        """确保当前线程在正确的 device 上下文中"""
        current = torch_npu.npu.current_device()
        if current != self.device:
            torch_npu.npu.set_device(self.device)

    # -------- tensor 注册 --------
    def add_swap_tensor(self, tensor_id: int, tensor: torch.Tensor):
        # 验证 tensor 是否在正确的 device 上
        if tensor.device.index != self.device:
            raise RuntimeError(
                f"Tensor device {tensor.device} mismatch with "
                f"SwapManager device {self.device}"
            )
        
        # 确保在正确的 device 上下文中操作
        self._ensure_device_context()
        self.swap_tensors[tensor_id] = SwapTensor(tensor)

    def get_swap_tensor(self, tensor_id: int):
        return self.swap_tensors.get(tensor_id, None)

    # -------- D2H --------
    def launch_d2h(self, tensor_id: int, stream):
        st = self.swap_tensors[tensor_id]
        if st.stat != "device":
            print(f"[DEBUG] Tensor {tensor_id} not in device state for D2H!")
            return

        # 验证 stream 的 device 匹配
        if hasattr(stream, 'device') and stream.device.index != self.device:
            raise RuntimeError(
                f"Stream device {stream.device} != manager device {self.device}"
            )

        self._ensure_device_context()
        
        evt = self.event_pool.acquire()
        st.d2h_event = evt

        with torch.no_grad():
            # 确保在正确的 device 和 stream 上下文中
            with torch.npu.device(self.device):
                with torch_npu.npu.stream(stream):
                    if st.is_slice_tensor:
                        st.tensor_cpu.copy_(st.tensor, non_blocking=True)
                    else:
                        st.tensor_cpu.storage().copy_(
                            st.tensor.storage(), non_blocking=True
                        )
                    evt.record(stream)

        st.stat = "d2h_inflight"

    def wait_d2h_finished(self, tensor_id: int, compute_stream=None):
        """
        D2H 是 host 侧关心的问题：
        - 必须确保 CPU buffer 写完
        - 之后才能 resize device storage
        """
        st = self.swap_tensors[tensor_id]
        if st.stat != "d2h_inflight":
            print(f"[DEBUG] Tensor {tensor_id} not in d2h_inflight state for wait!")
            return

        self._ensure_device_context()

        # 关键修正点：host-side 同步
        if compute_stream is not None and st.d2h_event is not None:
            compute_stream.wait_event(st.d2h_event)
        if st.d2h_event is not None:
            self.event_pool.release(st.d2h_event)
            st.d2h_event = None

        # 释放 device storage（危险但受控）
        with torch.npu.device(self.device):
            st.tensor.storage().resize_(0)

        st.stat = "host"

    # -------- H2D --------
    def launch_h2d(self, tensor_id: int, stream):
        st = self.swap_tensors[tensor_id]
        if st.stat != "host":
            print(f"[DEBUG] Tensor {tensor_id} not in host state for H2D!")
            return

        # 验证 stream 的 device 匹配
        if hasattr(stream, 'device') and stream.device.index != self.device:
            raise RuntimeError(
                f"Stream device {stream.device} != manager device {self.device}"
            )

        self._ensure_device_context()

        # 先恢复 storage
        with torch.npu.device(self.device):
            st.tensor.storage().resize_(st.storage_size)

        evt = self.event_pool.acquire()
        st.h2d_event = evt

        with torch.no_grad():
            with torch.npu.device(self.device):
                with torch_npu.npu.stream(stream):
                    if st.is_slice_tensor:
                        st.tensor.copy_(st.tensor_cpu, non_blocking=True)
                    else:
                        st.tensor.storage().copy_(
                            st.tensor_cpu.storage(), non_blocking=True
                        )
                    evt.record(stream)

        st.stat = "h2d_inflight"

    def wait_h2d_finished(self, tensor_id: int, compute_stream):
        """
        H2D 的 wait 必须发生在消费该 tensor 的 compute stream 上
        """
        st = self.swap_tensors[tensor_id]
        if st.stat != "h2d_inflight":
            print(f"[DEBUG] Tensor {tensor_id} not in h2d_inflight state for wait!")
            return

        # 验证 stream 的 device 匹配
        if hasattr(compute_stream, 'device') and compute_stream.device.index != self.device:
            raise RuntimeError(
                f"Compute stream device {compute_stream.device} != manager device {self.device}"
            )

        self._ensure_device_context()

        if st.h2d_event is not None:
            with torch.npu.device(self.device):
                compute_stream.wait_event(st.h2d_event)
                self.event_pool.release(st.h2d_event)
                st.h2d_event = None

        st.stat = "device"

    # -------- batch / epoch 结束清理 --------
    def clear(self):
        self._ensure_device_context()
        
        for st in self.swap_tensors.values():
            st.tensor_cpu = None
            st.d2h_event = None
            st.h2d_event = None
        self.swap_tensors.clear()

    def is_h2d_finished(self, tensor_id: int):
        """
        检查 H2D 是否完成（非阻塞）
        """
        st = self.swap_tensors.get(tensor_id, None)
        if st is None:
            print(f"[DEBUG] Tensor {tensor_id} not registered")
            return False

        if st.stat != "h2d_inflight":
            return True

        if st.h2d_event is None:
            print(f"[DEBUG] Tensor {tensor_id} has no h2d_event!")
            return False

        self._ensure_device_context()
        finished = st.h2d_event.query()
        if not finished:
            print(f"[DEBUG] Tensor {tensor_id} H2D NOT finished yet!")
        return finished

    def debug_h2d_all(self):
        """
        打印所有 swap tensor H2D 状态
        """
        self._ensure_device_context()
        for tid, st in self.swap_tensors.items():
            status = st.stat
            finished = st.h2d_event.query() if st.h2d_event else True
            print(f"[DEBUG] Tensor {tid}: stat={status}, h2d_finished={finished}")