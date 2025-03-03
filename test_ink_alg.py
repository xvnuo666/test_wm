import torch
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Device
# device = "cuda" if torch.cuda.is_available() else "cpu"
device=('cpu')
model_name='/home/buhaoran2023/LLM_Models/Meta/Llama2/Llama-2-7b-hf'
# Transformers config
transformers_config = TransformersConfig(model=AutoModelForCausalLM.from_pretrained(model_name).to(device),
                                         tokenizer=AutoTokenizer.from_pretrained(model_name),
                                         vocab_size=50272,
                                         device=device,
                                         max_new_tokens=200,
                                         min_length=230,
                                         do_sample=True,
                                         no_repeat_ngram_size=4)

# Load watermark algorithm
myWatermark = AutoWatermark.load('KGW',
                                 algorithm_config='config/KGW.json',
                                 transformers_config=transformers_config)

# Prompt
prompt = 'Good Morning.'

# Generate and detect
watermarked_text = myWatermark.generate_watermarked_text(prompt)
detect_result = myWatermark.detect_watermark(watermarked_text)
unwatermarked_text = myWatermark.generate_unwatermarked_text(prompt)
detect_result = myWatermark.detect_watermark(unwatermarked_text)