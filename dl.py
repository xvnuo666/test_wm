# from transformers import AutoModel, AutoTokenizer
#
# # 指定模型名称和保存目录
# model_name = "facebook/opt-1.3b"
# save_directory = "/home/xunuo2024/xn_code/opt"
#
# # 下载并保存分词器
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.save_pretrained(save_directory)
#
# # 下载并保存模型
# model = AutoModel.from_pretrained(model_name)
# model.save_pretrained(save_directory)



# from transformers import AutoModel, AutoTokenizer
# import os
#
# # 指定模型名称和保存目录
# model_name = "perceptiveshawty/compositional-bert-large-uncased"
# save_directory = "watermark/sir/model/"
#
# # 确保保存目录存在
# os.makedirs(save_directory, exist_ok=True)
#
# # 下载并保存分词器
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.save_pretrained(save_directory)
#
# # 下载并保存模型
# model = AutoModel.from_pretrained(model_name)
# model.save_pretrained(save_directory)


from transformers import AutoModel, AutoTokenizer

# 定义模型名称（需确认模型在 Hugging Face Hub 上的准确名称）
model_name = "perceptiveshawty/compositional-bert-large-uncased"  # 请确保名称正确

# 下载并加载模型与分词器
try:
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("模型与分词器下载成功！")
except Exception as e:
    print(f"下载失败，请检查模型名称或网络连接。错误信息：{e}")

# 保存到本地目录（可选）
save_dir = "/home/xunuo2024/xn_code/markllm/watermark/sir/model/compositional-bert-large-uncased"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"模型已保存至 {save_dir}")