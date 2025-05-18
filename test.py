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

def showInfo():
    if torch.cuda.is_available():
        cuda_device = torch.cuda.get_device_name(0)
    print(cuda_device)

from enum import Enum, auto

class Color(Enum):
    RED = auto()
    GREEN = auto()
    BLUE = auto()

    def __str__(self):
        match self:
            case Color.RED:
                return "<RED>"
            case Color.GREEN:
                return "<GREEN>"
            case Color.BLUE:
                return "<BLUE>"

# 使用枚举
def draw(color: Color):
    print(f"Drawing {color}")

print(isinstance(Color.RED, Color))
draw(Color.RED)  # 明确、可读

def yeild_func(id, group, count):
    print(id , group , count)

inputs = {
    "id" : 111,
    "group": "nihapo",
    "count": 1
}

yeild_func(**inputs)

aa = torch.rand([1,20])
bb = torch.randn([1,21])
print(len(aa[0]))
print(bb[0,-1].item())
print(bb.tolist()[0][len(aa[0]):])

