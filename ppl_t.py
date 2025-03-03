import torch
from evaluation.dataset import C4Dataset
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluation.tools.text_editor import TruncatePromptTextEditor
from evaluation.tools.text_quality_analyzer import PPLCalculator
from evaluation.pipelines.quality_analysis import DirectTextQualityAnalysisPipeline, QualityPipelineReturnType
import os
# Load dataset
my_dataset = C4Dataset('dataset/c4/processed_c4.json')

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Transformer config
transformers_config = TransformersConfig(
    model=AutoModelForCausalLM.from_pretrained('facebook/opt-1.3b').to(device),                             	tokenizer=AutoTokenizer.from_pretrained('facebook/opt-1.3b'),
    vocab_size=50272,
    device=device,
    max_new_tokens=200,
    min_length=230,
    do_sample=True,
    no_repeat_ngram_size=4)

# Load watermark algorithm
my_watermark = AutoWatermark.load('EXPGumbel',
                                  algorithm_config='config/EXPGumbel.json',
                                  transformers_config=transformers_config)

# Init pipeline
quality_pipeline = DirectTextQualityAnalysisPipeline(
    dataset=my_dataset,
    watermarked_text_editor_list=[TruncatePromptTextEditor()],
    unwatermarked_text_editor_list=[],
    analyzer=PPLCalculator(
        model=AutoModelForCausalLM.from_pretrained('/home/buhaoran2023/LLM_Models/Meta/Llama2/Llama-2-7b-hf', device_map='auto'),
        tokenizer=AutoTokenizer.from_pretrained('/home/buhaoran2023/LLM_Models/Meta/Llama2/Llama-2-7b-hf'),
        device=device),
    unwatermarked_text_source='natural',
    show_progress=True,
    return_type=QualityPipelineReturnType.MEAN_SCORES)

# Evaluate
print(quality_pipeline.evaluate(my_watermark))