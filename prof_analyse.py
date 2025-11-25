from torch_npu.profiler.profiler import analyse
if __name__ == "__main__":
    analyse(profiler_path="/home/user2/ResNet_pytorch/export_only_prof_dir/train05_1824760_20251118051549531_ascend_pt", max_process_number=16)