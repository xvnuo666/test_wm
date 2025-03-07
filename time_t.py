import torch
from evaluation.dataset import C4Dataset
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluation.tools.text_editor import TruncatePromptTextEditor
from evaluation.tools.text_quality_analyzer import PPLCalculator
from evaluation.pipelines.quality_analysis import DirectTextQualityAnalysisPipeline, QualityPipelineReturnType
import os
import time
import json
# Load data
with open('dataset/c4/processed_c4.json', 'r') as f:
    lines = f.readlines()
    lines = [json.loads(line) for line in lines]

# # Load dataset
# my_dataset = C4Dataset('dataset/c4/processed_c4.json')

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = '/home/buhaoran2023/LLM_Models/Meta/Llama2/Llama-2-7b-hf'
# model_name='/home/buhaoran2023/LLM_Models/Microsoft/Phi-3-mini-4k-instruct'
# Transformer config
model=AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float16).to(device)
tokenizer=AutoTokenizer.from_pretrained(model_name)
print("len(tokenizer)",len(tokenizer))
print("tokenizer.vocab_size",tokenizer.vocab_size)
def cal_time(algorithm_name,test_len):
    transformers_config = TransformersConfig(
        model=model,
        tokenizer=tokenizer,
        # vocab_size=tokenizer.vocab_size,
        vocab_size=32000,
        device=device,
        max_new_tokens=200,
        min_length=230,
        do_sample=False,
        # do_sample=True,
        temperature=1,
        no_repeat_ngram_size=4)
    # Load watermark algorithm
    my_watermark = AutoWatermark.load(algorithm_name,
                                      algorithm_config=f'config/{algorithm_name}.json',
                                      transformers_config=transformers_config)

    print(lines[0])
    # test_len=2
    # time1=os.times()
    start_generate = time.perf_counter()
    text_list=[]
    untext_list=[]
    for i in range(test_len):
        text=my_watermark.generate_watermarked_text(prompt=lines[i]['prompt'],max_new_tokens=200)
        # print("text",text)
        text_list.append(text)
        untext = my_watermark.generate_unwatermarked_text(prompt=lines[i]['prompt'],max_new_tokens=200)
        # print("untext", untext)
        untext_list.append(untext)
    # time2=os.times()
    end_generate = time.perf_counter()
    print("time_generate:",end_generate-start_generate)
    start_detect = time.perf_counter()
    for i in range(test_len):
        # Detection
        detect_result_watermarked = my_watermark.detect_watermark(text)
        # print(detect_result_watermarked)
        detect_result_unwatermarked = my_watermark.detect_watermark(untext)
        # print(detect_result_unwatermarked)
    end_detect = time.perf_counter()
    print("time_detect:",end_detect-start_detect)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='KGW')
    parser.add_argument('--test_len', type=int, default=1)
    args = parser.parse_args()

    cal_time(args.algorithm, args.test_len)