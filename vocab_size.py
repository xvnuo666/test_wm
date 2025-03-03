from transformers import AutoTokenizer

model_name='/home/buhaoran2023/LLM_Models/Meta/Llama2/Llama-2-7b-hf'
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 查看词汇表大小
print(f"Vocabulary size: {len(tokenizer)}")