import torch# 打印可用 GPU 数量
print(f"可用 GPU 数量: {torch.cuda.device_count()}")

# 打印 GPU 名称
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")