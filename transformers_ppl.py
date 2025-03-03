import os.path
import numpy as np

from visualize.color_scheme import ColorSchemeForDiscreteVisualization
from visualize.font_settings import FontSettings
from visualize.legend_settings import DiscreteLegendSettings
from visualize.page_layout_settings import PageLayoutSettings
from visualize.visualizer import DiscreteVisualizer
from vllm import LLM, SamplingParams
import torch
from watermark.auto_watermark import AutoWatermark
import gc
import sys
import json
from watermark.auto_watermark import AutoWatermarkForVLLM
from utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Clean gpu memory
assert torch.cuda.is_available()
gc.collect()
torch.cuda.empty_cache()
with torch.no_grad():
    torch.cuda.empty_cache()

# Load data
with open('dataset/c4/processed_c4.json', 'r') as f:
    lines = f.readlines()
    lines = [json.loads(line) for line in lines]


def main(algorithm_name, model_path):
    config = AutoConfig.from_pretrained(model_path)
    model_for_causalllm = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Check if the tokenizer has a pad token, if not, set it to eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_t = next(model_for_causalllm.parameters()).device
    print(f"Model is loaded on device: {device_t}")
    transformers_config = TransformersConfig(
        model=model_for_causalllm,
        tokenizer=tokenizer,
        vocab_size=config.vocab_size,
        device="cuda",
        max_new_tokens=256,
        max_length=256,
        do_sample=True,
        no_repeat_ngram_size=4
    )
    watermark = AutoWatermark.load(algorithm_name=algorithm_name, algorithm_config=f'config/{algorithm_name}.json',
                                   transformers_config=transformers_config)
    visualizer = DiscreteVisualizer(color_scheme=ColorSchemeForDiscreteVisualization(),
                                    font_settings=FontSettings(),
                                    page_layout_settings=PageLayoutSettings(),
                                    legend_settings=DiscreteLegendSettings())

    prompts = [line['prompt'] for line in lines]
    prompts = prompts[:100]
    references = [line['natural_text'] for line in lines]

    generate_kwargs = {
        "max_length": 256,
        "min_length": 16,
        "do_sample": True,
        "temperature": 1.0,
        "num_return_sequences": 1,
        "logits_processor": [],
        "output_scores": True,  # 获取生成过程中的分数
        "return_dict_in_generate": True,  # 返回字典格式的结果
    }

    # Tokenize prompts
    input_ids = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device_t)

    # without watermark
    outputs = model_for_causalllm.generate(
        input_ids=input_ids,
        **generate_kwargs,
    )

    nowatermark_text = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
    nowatermark_cumulative_logprob = calculate_cumulative_logprob(outputs.scores, outputs.sequences)
    nowatermark_ppl = np.mean(
        [-logprob / len(seq) for logprob, seq in zip(nowatermark_cumulative_logprob, outputs.sequences)])
    nowatermark_detect_results = np.mean([r['is_watermarked'] for r in watermark.detect_watermark(nowatermark_text)])
    print(f"nowatermark_ppl: {nowatermark_ppl:.3f}")
    print(f"nowatermark_detect_results: {nowatermark_detect_results:.3f}")

    # with watermark
    watermark_processor = watermark.get_watermark_processor()
    generate_kwargs_with_watermark = generate_kwargs.copy()
    generate_kwargs_with_watermark["logits_processor"] = [watermark_processor]

    outputs = model_for_causalllm.generate(
        input_ids=input_ids,
        **generate_kwargs_with_watermark,
    )

    watermark_text = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
    watermark_cumulative_logprob = calculate_cumulative_logprob(outputs.scores, outputs.sequences)
    watermark_ppl = np.mean(
        [-logprob / len(seq) for logprob, seq in zip(watermark_cumulative_logprob, outputs.sequences)])
    watermark_detect_results = np.mean([r['is_watermarked'] for r in watermark.detect_watermark(watermark_text)])
    print(f"watermark_ppl: {watermark_ppl:.3f}")
    print(f"watermark_detect_results: {watermark_detect_results:.3f}")

    # # visualize
    # nowatermarked_img = visualizer.visualize(
    #     data=watermark.get_data_for_visualization(text=nowatermark_text[0]),
    #     show_text=True, visualize_weight=True, display_legend=True
    # )
    # nowatermarked_img.save(os.path.join(model_path, f"{algorithm_name}-nowatermark-vllm.png"))
    # watermarked_img = visualizer.visualize(
    #     data=watermark.get_data_for_visualization(text=watermark_text[0]),
    #     show_text=True, visualize_weight=True, display_legend=True
    # )
    # watermarked_img.save(os.path.join(model_path, f"{algorithm_name}-watermark-vllm.png"))


def calculate_cumulative_logprob(scores, sequences):
    cumulative_logprob = []
    for i, seq in enumerate(sequences):
        logprob = 0.0
        for t in range(1, len(seq)):
            token_id = seq[t].item()
            logit = scores[t - 1][i][token_id].item()
            logprob += logit
        cumulative_logprob.append(logprob)
    return cumulative_logprob


if __name__ == "__main__":
    # model_path = sys.argv[-2] # "meta-llama/Meta-Llama-3-8B-Instruct"
    # method = sys.argv[-1] # "UPV" "KGW" "Unigram"

    model_path = '/home/buhaoran2023/LLM_Models/Meta/Llama2/Llama-2-7b-hf'  # "meta-llama/Meta-Llama-3-8B-Instruct"
    method = "KGW"  # "UPV" "KGW" "Unigram"

    main(model_path=model_path, algorithm_name=method)
    """
    --------------------------------------------------------------
    llama3-8b-instruct (vLLM)
                           KGW               UPV           Unigram
    PPL         1.191 -> 1.346    1.191 -> 0.926    1.191 -> 1.344
    detect      0.001 -> 0.929    0.001 -> 0.430    0.001 -> 0.508
    time (h)      0.19 -> 0.52      0.18 -> 2.02      0.18 -> 0.45
    --------------------------------------------------------------
    llama3-8b-instruct (huggingface)
                           KGW               UPV           Unigram
    detect      0.001 -> 0.934    0.001 -> 0.358    0.001 -> 0.505
    time (h)    20.00 -> 20.75    19.50 -> 21.50    20.50 -> 20.50
    --------------------------------------------------------------
    """
