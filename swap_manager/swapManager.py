# swapManager.py：激进优化版 - 最小同步，最大重叠
import torch
import torch_npu
from typing import Dict, Optional


class EventPool:
    """Event 复用池"""
    def __init__(self, num_events=64, device=None):
        if device is None:
            device = torch_npu.npu.current_device()
        self.device = device
        self.pool = []
        with torch.npu.device(device):
            for _ in range(num_events):
                self.pool.append(torch.npu.Event(enable_timing=False))
        self.idx = 0

    def acquire(self):
        evt = self.pool[self.idx]
        self.idx = (self.idx + 1) % len(self.pool)
        return evt

    def release(self, evt):
        pass


class SwapTensor:
    """使用 pin_memory 加速传输"""
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor
        self.size = tensor.size()
        self.storage_size = tensor.storage().size()
        self.is_slice_tensor = tensor.storage().size() != tensor.numel()

        # pin_memory 加速 H2D
        if self.is_slice_tensor:
            self.tensor_cpu = torch.empty(
                tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=True
            )
        else:
            self.tensor_cpu = torch.empty(
                self.storage_size, dtype=tensor.dtype, device="cpu", pin_memory=True
            ).view(tensor.shape)

        self.stat = "device"
        self.d2h_event: Optional[torch.npu.Event] = None
        self.h2d_event: Optional[torch.npu.Event] = None
        self._pending_resize = False


class SwapManager:
    """激进优化：单一 prefetch_stream，最小同步"""

    def __init__(self, device=None, num_events: int = 64):
        if device is None:
            self.device = torch_npu.npu.current_device()
        else:
            self.device = device if isinstance(device, int) else device.index

        self.swap_tensors: Dict[int, SwapTensor] = {}

        with torch.npu.device(self.device):
            self.event_pool = EventPool(num_events=num_events, device=self.device)
            self.prefetch_stream = torch_npu.npu.Stream(device=self.device)
            self._cached_compute_stream: Optional[torch.npu.Stream] = None

    def _get_compute_stream(self):
        if self._cached_compute_stream is None:
            self._cached_compute_stream = torch_npu.npu.current_stream()
        return self._cached_compute_stream

    def add_swap_tensor(self, tensor_id: int, tensor: torch.Tensor):
        if tensor.device.index != self.device:
            raise RuntimeError(f"Device mismatch: {tensor.device} vs {self.device}")
        self.swap_tensors[tensor_id] = SwapTensor(tensor)

    def get_swap_tensor(self, tensor_id: int):
        return self.swap_tensors.get(tensor_id, None)

    def launch_d2h(self, tensor_id: int, compute_stream=None):
        """立即在后台启动 D2H，不阻塞"""
        st = self.swap_tensors[tensor_id]
        if st.stat != "device":
            return

        if compute_stream is None:
            compute_stream = self._get_compute_stream()

        # 让 prefetch_stream 等待计算就绪，然后启动传输
        ready_event = torch.npu.Event()
        ready_event.record(compute_stream)
        self.prefetch_stream.wait_event(ready_event)

        st.d2h_event = self.event_pool.acquire()

        with torch.no_grad():
            with torch_npu.npu.stream(self.prefetch_stream):
                if st.is_slice_tensor:
                    st.tensor_cpu.copy_(st.tensor, non_blocking=True)
                else:
                    st.tensor_cpu.storage().copy_(st.tensor.storage(), non_blocking=True)
                st.d2h_event.record(self.prefetch_stream)

        st.stat = "d2h_inflight"

    def wait_d2h_finished(self, tensor_id: int, compute_stream=None, need_wait: bool = False):
        """
        默认不等待（need_wait=False），让 D2H 在后台完成。
        仅在显式需要时同步（如内存紧张）。
        """
        st = self.swap_tensors[tensor_id]
        if st.stat != "d2h_inflight":
            return

        if compute_stream is None:
            compute_stream = self._get_compute_stream()

        if need_wait and st.d2h_event is not None:
            compute_stream.wait_event(st.d2h_event)

        if st.d2h_event is not None:
            self.event_pool.release(st.d2h_event)
            st.d2h_event = None

        # 延迟 resize
        if need_wait:
            with torch.npu.device(self.device):
                st.tensor.storage().resize_(0)
        else:
            st._pending_resize = True

        st.stat = "host"

    def launch_h2d(self, tensor_id: int, compute_stream=None, flag: bool = True):
        """在后台启动 H2D"""
        st = self.swap_tensors[tensor_id]
        if st.stat != "host":
            return

        if compute_stream is None:
            compute_stream = self._get_compute_stream()

        # 恢复 storage
        if flag:
            with torch.npu.device(self.device):
                st.tensor.storage().resize_(st.storage_size)

        backward_event = torch.npu.Event()
        backward_event.record(compute_stream)
        self.prefetch_stream.wait_event(backward_event)

        st.h2d_event = self.event_pool.acquire()

        with torch.no_grad():
            with torch_npu.npu.stream(self.prefetch_stream):
                if st.is_slice_tensor:
                    st.tensor.copy_(st.tensor_cpu, non_blocking=True)
                else:
                    st.tensor.storage().copy_(st.tensor_cpu.storage(), non_blocking=True)
                st.h2d_event.record(self.prefetch_stream)

        st.stat = "h2d_inflight"

    def wait_h2d_finished(self, tensor_id: int, compute_stream=None, need_wait: bool = True):
        """
        H2D 通常需要等待（数据要用了），但先查询避免不必要的阻塞。
        """
        st = self.swap_tensors[tensor_id]
        if st.stat != "h2d_inflight":
            return

        if compute_stream is None:
            compute_stream = self._get_compute_stream()

        if need_wait and st.h2d_event is not None:
            # 先查询，已完成则跳过 wait
            if not st.h2d_event.query():
                compute_stream.wait_event(st.h2d_event)
            self.event_pool.release(st.h2d_event)
            st.h2d_event = None

        st.stat = "device"

    def sync_all_d2h(self, tensor_ids: list, compute_stream=None, need_wait: bool = False):
        """批量同步 D2H"""
        if compute_stream is None:
            compute_stream = self._get_compute_stream()

        for tid in tensor_ids:
            st = self.swap_tensors.get(tid)
            if st and st.stat == "d2h_inflight":
                if need_wait and st.d2h_event is not None:
                    compute_stream.wait_event(st.d2h_event)
                if st.d2h_event is not None:
                    self.event_pool.release(st.d2h_event)
                    st.d2h_event = None
                st.stat = "host"
                if need_wait:
                    with torch.npu.device(self.device):
                        st.tensor.storage().resize_(0)
                else:
                    st._pending_resize = True

    def h2d_prefetch(self, tensor_ids: list, compute_stream=None):
        """批量预取 H2D"""
        if compute_stream is None:
            compute_stream = self._get_compute_stream()

        ready_event = torch.npu.Event()
        ready_event.record(compute_stream)
        self.prefetch_stream.wait_event(ready_event)

        for tid in tensor_ids:
            st = self.swap_tensors.get(tid)
            if st and st.stat == "host":
                with torch.npu.device(self.device):
                    st.tensor.storage().resize_(st.storage_size)

                with torch.no_grad():
                    with torch_npu.npu.stream(self.prefetch_stream):
                        if st.is_slice_tensor:
                            st.tensor.copy_(st.tensor_cpu, non_blocking=True)
                        else:
                            st.tensor.storage().copy_(st.tensor_cpu.storage(), non_blocking=True)

                st.stat = "h2d_inflight"

    def clear(self):
        for st in self.swap_tensors.values():
            st.tensor_cpu = None
            st.d2h_event = None
            st.h2d_event = None
        self.swap_tensors.clear()
        self._cached_compute_stream = None

    def is_h2d_finished(self, tensor_id: int) -> bool:
        st = self.swap_tensors.get(tensor_id)
        if st is None or st.stat != "h2d_inflight":
            return True
        if st.h2d_event is None:
            return True
        return st.h2d_event.query()

    def debug_status(self):
        for tid, st in self.swap_tensors.items():
            d2h_done = st.d2h_event.query() if st.d2h_event else True
            h2d_done = st.h2d_event.query() if st.h2d_event else True
            print(f"[DEBUG] Tensor {tid}: stat={st.stat}, d2h_done={d2h_done}, h2d_done={h2d_done}")