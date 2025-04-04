import torch

# 检查版本和路径
print(f"PyTorch 版本: {torch.__version__}")  
print(f"安装路径: {torch.__file__}")  

# 测试 GPU 支持
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 设备: {torch.cuda.get_device_name(0)}")

# 基础张量运算
x = torch.tensor([1, 2, 3])
print(x + 1)  # 应输出 tensor([2, 3, 4])