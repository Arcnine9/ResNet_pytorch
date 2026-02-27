from torch_npu.profiler.profiler import analyse
if __name__ == "__main__":
    analyse(profiler_path="/home/user8/ResNet_pytorch/train07_1627902_20260227070131106_ascend_pt", max_process_number=16)