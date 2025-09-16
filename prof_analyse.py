from torch_npu.profiler.profiler import analyse

if __name__ == "__main__":
    analyse(profiler_path="/data/train5_data/ResNet/profiling_data/profiling_data/train05_844416_20250916112348372_ascend_pt", max_process_number=16)