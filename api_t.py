
from fastapi import FastAPI, Request
import uvicorn, json, datetime

from visualize.color_scheme import ColorSchemeForDiscreteVisualization
from visualize.font_settings import FontSettings
from visualize.legend_settings import DiscreteLegendSettings
from visualize.page_layout_settings import PageLayoutSettings
from visualize.visualizer import DiscreteVisualizer
from evaluation.tools.text_quality_analyzer import PPLCalculator,LogDiversityAnalyzer
from vllm import LLM, SamplingParams

import gc
import sys
import json
import torch
from watermark.auto_watermark import AutoWatermarkForVLLM
from utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

model = None
opt = None
watermark=None
visualizer=None

# Clean gpu memory
assert torch.cuda.is_available()
gc.collect()
torch.cuda.empty_cache()
with torch.no_grad():
    torch.cuda.empty_cache()
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
model_path="/home/buhaoran2023/LLM_Models/Meta/Llama2/Llama-2-7b-hf"       #模型的地址
model_name="meta-llama/Meta-Llama-3-8B-Instruct"
def load_model():
    global model, config, transformers_config
    model = LLM(
        model=model_path, trust_remote_code=True,
        max_model_len=256,
        gpu_memory_utilization=0.9,
        enforce_eager=False,
        dtype="half",
        disable_custom_all_reduce=False,
        disable_log_stats=False,
        swap_space=32,
        seed=42
    )
    config = AutoConfig.from_pretrained(model_path)
    transformers_config = TransformersConfig(
        model=AutoModelForCausalLM.from_pretrained(model_path),
        trust_remote_code=True,
        tokenizer=AutoTokenizer.from_pretrained(model_path),
        vocab_size=config.vocab_size,
        device="cuda",
        max_new_tokens=256,
        max_length=256,
        do_sample=True,
        no_repeat_ngram_size=4
    )

app = FastAPI()
algorithm_name = "KGW"
@app.on_event("startup")
async def startup_event():
    load_model()
    watermark = AutoWatermarkForVLLM(algorithm_name=algorithm_name, algorithm_config=f'config/{algorithm_name}.json',
                                     transformers_config=transformers_config)
    visualizer = DiscreteVisualizer(color_scheme=ColorSchemeForDiscreteVisualization(),
                                    font_settings=FontSettings(),
                                    page_layout_settings=PageLayoutSettings(),
                                    legend_settings=DiscreteLegendSettings())
    return watermark, visualizer

@app.post("/watermark_generate")
async def water_generate(prompts:str):
    # without watermark
    outputs = model.generate(
        prompts=prompts,
        sampling_params=SamplingParams(
            n=1, temperature=1.0, seed=42,
            max_tokens=256, min_tokens=16,
            logits_processors=[],
            logprobs=True
        ),
        use_tqdm=True,
    )
    nowatermark_text = [output.outputs[0].text for output in outputs]
    # with watermark
    outputs = model.generate(
        prompts=prompts,
        sampling_params=SamplingParams(
            n=1, temperature=1.0, seed=42,
            max_tokens=256, min_tokens=16,
            logits_processors=[watermark],
            logprobs=True
        ),
        use_tqdm=True,
    )
    watermark_text = [output.outputs[0].text for output in outputs]
    return watermark_text,nowatermark_text

import random
@app.post("/del_text")
async def del_text(text:str,ratio:float):
    """Delete words randomly from the text."""

    # Handle empty string input
    if not text:
        return text
    # Split the text into words and randomly delete each word based on the ratio
    word_list = text.split()
    edited_words = [word for word in word_list if random.random() >= ratio]

    # Join the words back into a single string
    deleted_text = ' '.join(edited_words)

    return deleted_text

@app.post("/detect_watermark")
async def detect_watermark(text:str):
    return watermark.detect_watermark(text)

@app.post("/quality_test")
async def quality_test(text:str):

    ppl_analyzer = PPLCalculator(
        model=AutoModelForCausalLM.from_pretrained(model_path,device_map='auto'),
        tokenizer=AutoTokenizer.from_pretrained(model_path),
        device=device)
    log_analyzer=LogDiversityAnalyzer()
    return ppl_analyzer.analyze(text), log_analyzer.analyze(text)

@app.post("/visualize")
async def visualize(text: str):
    img = visualizer.visualize(
        data=watermark.get_data_for_visualization(text=text),
        show_text=True, visualize_weight=True, display_legend=True
    )
    return img
if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=10001, workers=1)
