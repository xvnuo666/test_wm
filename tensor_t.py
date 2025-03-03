import numpy as np
import torch

torch.manual_seed(36)  # 可以选择任意整数作为种子

# 使用 PyTorch 直接创建一个 6x6 的张量
torch_tensor = torch.rand(6, 6)  # 默认情况下，rand 函数生成 float32 类型的张量

print(torch_tensor)
print(np.arange(3))
print(torch_tensor[[1,3]])


# import torch
#
# # 设置 PyTorch 的随机种子
# torch.manual_seed(42)  # 可以选择任意整数作为种子
#
# # 如果你正在使用GPU，你也应该设置CUDA的随机种子
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(42)
#     torch.cuda.manual_seed_all(42)  # 如果使用多个GPU
#
# # 设置其他可能影响结果的选项（可选）
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
#
# # 使用 PyTorch 直接创建一个 6x6 的张量
# torch_tensor = torch.rand(6, 6)
#
# print(torch_tensor)