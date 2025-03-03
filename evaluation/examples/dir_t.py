import os
from evaluation.dataset import C4Dataset

# 获取当前脚本的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 构建数据文件的绝对路径
data_path = os.path.join(script_dir, '..', '..', 'dataset', 'c4', 'processed_c4.json')

# 使用绝对路径创建 C4Dataset 实例
my_dataset = C4Dataset(data_path)
print(data_path)
print(script_dir)