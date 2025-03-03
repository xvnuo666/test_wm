import torch

import numpy as np
# from markllm.watermark.auto_watermark import AutoWatermark
# from markllm.utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name='/home/buhaoran2023/LLM_Models/Meta/Llama2/Llama-2-7b-hf'
# model_name='/home/buhaoran2023/LLM_Models/Meta/Llama3/Meta-Llama-3-8B'
model=AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer=AutoTokenizer.from_pretrained(model_name)

text = 'Good Morning.'
criterion = torch.nn.CrossEntropyLoss()
print(text)
encoded_text = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)
print(encoded_text)
# logits = model(torch.unsqueeze(encoded_text, 0), return_dict=True).logits[0]
logits = model(torch.unsqueeze(encoded_text, 0), return_dict=True).logits[0]
print(logits)
loss = criterion(logits[:-1], encoded_text[1:])
ppl = torch.exp(loss)