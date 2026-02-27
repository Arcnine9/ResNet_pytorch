import torch
import torch_npu

device = "npu:0"

x = torch.randn(2, 3, 224, 224, device=device)
conv = torch.nn.Conv2d(3, 64, 7, stride=2, padding=3).to(device)

y = conv(x)
print(y.shape)
