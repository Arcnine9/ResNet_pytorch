import torchvision
import os

# 创建数据目录
data_root = "/data/train8_data/DNN_data"
os.makedirs(data_root, exist_ok=True)

# 下载 CIFAR-10（自动处理 train/test 分割）
print("Downloading CIFAR-10...")
cifar10_train = torchvision.datasets.CIFAR10(
    root=data_root, 
    train=True, 
    download=True
)
cifar10_test = torchvision.datasets.CIFAR10(
    root=data_root, 
    train=False, 
    download=True
)

print(f"CIFAR-10 训练集: {len(cifar10_train)} 张")
print(f"CIFAR-10 测试集: {len(cifar10_test)} 张")
print(f"类别: {cifar10_train.classes}")
print(f"保存路径: {os.path.abspath(data_root)}/cifar-10-batches-py")