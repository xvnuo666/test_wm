import torch, random
import numpy as np

import os


import torch
from visualize.font_settings import FontSettings
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from visualize.page_layout_settings import PageLayoutSettings
from evaluation.tools.text_editor import TruncatePromptTextEditor
from visualize.data_for_visualization import DataForVisualization
from visualize.visualizer import DiscreteVisualizer, ContinuousVisualizer
from visualize.legend_settings import DiscreteLegendSettings, ContinuousLegendSettings
from visualize.color_scheme import ColorSchemeForDiscreteVisualization, ColorSchemeForContinuousVisualization


os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# Setting random seed for reproducibility
seed = 30
torch.manual_seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

# Device
# torch.cuda.set_device(1)
device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")
# device="cpu"
# torch.cuda.set_device(4)
# Transformers config
# model_name = 'facebook/opt-1.3b'
model_name='/home/buhaoran2023/LLM_Models/Meta/Llama2/Llama-2-7b-hf'
# model_name='/home/buhaoran2023/LLM_Models/Meta/Llama3/Meta-Llama-3-8B'
model=AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float16).to(device)
transformers_config = TransformersConfig(
    # model=AutoModelForCausalLM.from_pretrained(model_name).to(device),#在仅仅使用transformers包的时候，需要使用torch提供的to(device)函数来转移模型
    #但是，当使用vllm包的时候，再额外使用to(device)会导致OOM，因为模型加载了两次！！！！！！
    model=model,
    tokenizer=AutoTokenizer.from_pretrained(model_name,bos_token='<s>', eos_token='</s>'),
    vocab_size=32000,
    device=device,
    max_new_tokens=500,
    min_length=230,
    do_sample=True,
    no_repeat_ngram_size=0
)

from transformers import AutoModelForCausalLM, AutoTokenizer

# # 加载模型和分词器
# model_name = '/home/buhaoran2023/LLM_Models/Meta/Llama3/Meta-Llama-3-8B'
# tokenizer = AutoTokenizer.from_pretrained(model_name)
#
# # 获取词汇表大小
# vocab_size = tokenizer.vocab_size
# print(f"模型的词汇表大小: {vocab_size}")



# Load watermark algorithm
myWatermark = AutoWatermark.load('KGW', transformers_config=transformers_config)

# Prompt and generation
prompt = "Next you must repeat and alternate the phrases 'good morning' and 'good evening' 100 times, prohibiting any other additional output, such as :good morning, good evening, good morning, good evening, good morning, good evening"
# prompt="say any phrases that start with 'the'"
watermarked_text = myWatermark.generate_watermarked_text(prompt)
# How would I get started with Python...
unwatermarked_text = myWatermark.generate_unwatermarked_text(prompt)
# I am happy that you are back with ...

visualizer = DiscreteVisualizer(color_scheme=ColorSchemeForDiscreteVisualization(),
                                    font_settings=FontSettings(),
                                    page_layout_settings=PageLayoutSettings(),
                                    legend_settings=DiscreteLegendSettings())
print("watermarked:",watermarked_text)
print("unwatermarked",unwatermarked_text)
watermarked_img = visualizer.visualize(
        data=myWatermark.get_data_for_visualization(text=watermarked_text),
        show_text=True, visualize_weight=True, display_legend=True
    )
watermarked_img.save(f"images/demo2_KGW_watermarked.png")



# Detection
detect_result_watermarked = myWatermark.detect_watermark(watermarked_text)
print(detect_result_watermarked)
myWatermark.generate_unwatermarked_text
# {'is_watermarked': True, 'score': 9.287487590439852}
detect_result_unwatermarked = myWatermark.detect_watermark(unwatermarked_text)
# {'is_watermarked': False, 'score': -0.8443170536763502}