# Pre-Profiling Module

神经网络模块预分析工具，用于测量各层前向/反向传播时间，支持DeepUM分块运行优化。

## 核心功能

1. **自动解析layer.config**: 从离线分析器输出提取shape和连接信息
2. **精确时间测量**: 支持warmup、多次测量、标准差计算
3. **多设备支持**: CPU、CUDA、NPU (Ascend)
4. **双向测量**: 同时测量前向和反向传播
5. **灵活输出**: 文本和JSON格式

## 快速开始

```python
from pre_profiling import run_pre_profiling
from torchvision.models import inception_v3

model = inception_v3(aux_logits=False)
results = run_pre_profiling(
    model=model,
    layer_config_path='./layers.config',  # 来自离线分析器
    output_path='module_profiling.txt',
    device='npu:0',
    num_runs=5,
    warmup=2
)