import torch
import torch_npu
from typing import Dict


# =========================
# EventPool：Event 复用池
# =========================
class EventPool:
    """
    简单环形复用池：
    - 不做引用计数
    - 假设调用方保证：同一个 event 不会在未完成时被复用
    """
    def __init__(self, num_events=128, device=None):
        self.device = device
        self.pool = []
        with torch.npu.device(device):
            for _ in range(num_events):
                self.pool.append(torch.npu.Event())
        self.idx = 0

    def acquire(self):
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

        # 状态机：
        # device | d2h_inflight | host | h2d_inflight
        self.stat = "device"

        # event 只是句柄
        self.d2h_event = None
        self.h2d_event = None


# =========================
# SwapManager：唯一调度者
# =========================
class SwapManager:
    def __init__(self, num_events: int = 128):
        self.swap_tensors: Dict[int, SwapTensor] = {}
        self.device = torch_npu.npu.current_device()
        self.event_pool = EventPool(num_events=num_events, device=self.device)

    # -------- tensor 注册 --------
    def add_swap_tensor(self, tensor_id: int, tensor: torch.Tensor):
        # if tensor_id in self.swap_tensors:
        #     return
        self.swap_tensors[tensor_id] = SwapTensor(tensor)

    def get_swap_tensor(self, tensor_id: int):
        return self.swap_tensors.get(tensor_id, None)

    # -------- D2H --------
    def launch_d2h(self, tensor_id: int, stream):
        st = self.swap_tensors[tensor_id]
        if st.stat != "device":
            print(f"[DEBUG] Tensor {tensor_id} not in device state for D2H!")
            return

        evt = self.event_pool.acquire()
        st.d2h_event = evt

        with torch.no_grad():
            with torch_npu.npu.stream(stream):
                if st.is_slice_tensor:
                    st.tensor_cpu.copy_(st.tensor, non_blocking=True)
                else:
                    st.tensor_cpu.storage().copy_(
                        st.tensor.storage(), non_blocking=True
                    )
                evt.record(stream)

        st.stat = "d2h_inflight"

    def wait_d2h_finished(self, tensor_id: int, compute_stream = None):
        """
        D2H 是 host 侧关心的问题：
        - 必须确保 CPU buffer 写完
        - 之后才能 resize device storage
        """
        st = self.swap_tensors[tensor_id]
        if st.stat != "d2h_inflight":
            print(f"[DEBUG] Tensor {tensor_id} not in d2h_inflight state for wait!")
            return

        # 关键修正点：host-side 同步
        if compute_stream is not None and st.d2h_event is not None:
           compute_stream.wait_event(st.d2h_event)
        if st.d2h_event is not None:
            self.event_pool.release(st.d2h_event)
            st.d2h_event = None

        # 释放 device storage（危险但受控）
        st.tensor.storage().resize_(0)

        st.stat = "host"

    # -------- H2D --------
    def launch_h2d(self, tensor_id: int, stream):
        st = self.swap_tensors[tensor_id]
        if st.stat != "host":
            print(f"[DEBUG] Tensor {tensor_id} not in host state for H2D!")
            return

        # 先恢复 storage
        st.tensor.storage().resize_(st.storage_size)

        evt = self.event_pool.acquire()
        st.h2d_event = evt

        with torch.no_grad():
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

        if st.h2d_event is not None:
            # 等待 compute stream 上事件
            compute_stream.wait_event(st.h2d_event)

            self.event_pool.release(st.h2d_event)
            st.h2d_event = None

        st.stat = "device"

    # -------- batch / epoch 结束清理 --------
    def clear(self):

        # 清理 CPU buffer 和事件
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
            # 已经完成或不在 H2D 状态
            return True

        if st.h2d_event is None:
            print(f"[DEBUG] Tensor {tensor_id} has no h2d_event!")
            return False

        finished = st.h2d_event.query()
        if not finished:
            print(f"[DEBUG] Tensor {tensor_id} H2D NOT finished yet!")
        return finished

    def debug_h2d_all(self):
        """
        打印所有 swap tensor H2D 状态
        """
        for tid, st in self.swap_tensors.items():
            status = st.stat
            finished = st.h2d_event.query() if st.h2d_event else True
            print(f"[DEBUG] Tensor {tid}: stat={status}, h2d_finished={finished}")
