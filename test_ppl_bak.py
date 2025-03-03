import torch
from evaluation.dataset import C4Dataset
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluation.tools.text_editor import TruncatePromptTextEditor
from evaluation.tools.text_quality_analyzer import PPLCalculator
from evaluation.pipelines.quality_analysis import DirectTextQualityAnalysisPipeline, QualityPipelineReturnType
import os
import json
from sentence_transformers import SentenceTransformer
def calculate_sbert_similarity(sentences, model_path):
    # 加载本地模型
    model = SentenceTransformer(model_path)
    cossim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    # 生成句向量
    embeddings = model.encode(sentences, convert_to_tensor=True)

    # 计算余弦相似度
    cosine_scores = cossim(embeddings[0], embeddings[1])
    print("cosine_scores",cosine_scores)
    return cosine_scores
    # # 打印相似度矩阵
    # for i in range(len(sentences)):
    #     for j in range(i + 1, len(sentences)):
    #         print(f"「{sentences[i]}」vs「{sentences[j]}」相似度: {cosine_scores[i][j]:.4f}")

# Load data
with open('dataset/c4/processed_c4.json', 'r') as f:
    lines = f.readlines()
    lines = [json.loads(line) for line in lines]

# Load dataset
my_dataset = C4Dataset('dataset/c4/processed_c4.json')

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model_name = '/home/buhaoran2023/LLM_Models/Meta/Llama2/Llama-2-7b-hf'
model_name='/home/buhaoran2023/LLM_Models/Microsoft/Phi-3-mini-4k-instruct'
# Transformer config
model=AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float16).to(device)
tokenizer=AutoTokenizer.from_pretrained(model_name)
print("len(tokenizer)",len(tokenizer))
print("tokenizer.vocab_size",tokenizer.vocab_size)
transformers_config = TransformersConfig(
    model=model,
    tokenizer=tokenizer,
    # vocab_size=tokenizer.vocab_size,
    vocab_size=32064,
    device=device,
    max_new_tokens=200,
    min_length=230,
    do_sample=False,
    # do_sample=True,
    no_repeat_ngram_size=4)

# Load watermark algorithm
my_watermark = AutoWatermark.load('DIP',
                                  algorithm_config='config/DIP.json',
                                  transformers_config=transformers_config)
# analyzer=PPLCalculator(
#         model=AutoModelForCausalLM.from_pretrained('..model/llama-7b/', device_map='auto'),                 		tokenizer=LlamaTokenizer.from_pretrained('..model/llama-7b/'),
#         device=device)
print(lines[0])
p1=0
p2=0
cos=0.0
test_len=10
for i in range(test_len):
    text=my_watermark.generate_watermarked_text(prompt=lines[i]['prompt'],
                                                 max_new_tokens=200,
                                                 min_length=230,
                                                 do_sample=True,
                                                 no_repeat_ngram_size=4)
    # inputs=lines[i]['prompt']+lines[i]['natural_text']
    inputs = lines[i]['prompt']
    inputs = transformers_config.tokenizer(inputs, return_tensors='pt').to(device)
    #  inputs=transformers_config.tokenizer(inputs, return_tensors='pt').to(device)
    unwatermark_output = model.generate(**inputs, max_new_tokens=200, min_length=230, do_sample=True,
                                        no_repeat_ngram_size=4)
    generated_text = transformers_config.tokenizer.decode(unwatermark_output[0], skip_special_tokens=True)
    print(generated_text)
    inputs = transformers_config.tokenizer(generated_text, return_tensors='pt').to(device)
    print("inputs",inputs)
    loss = model(input_ids=inputs.input_ids,  labels=inputs.input_ids).loss
    perplexity1 = torch.exp(loss)   # 直接通过交叉熵损失计算困惑度
    print("P1",perplexity1)
    p1=p1+perplexity1.item()
    text=transformers_config.tokenizer(text, return_tensors='pt').to(device)
    print("text",text)
    loss=model(input_ids=text.input_ids, labels=text.input_ids).loss
    perplexity2 = torch.exp(loss)
    print("P2",perplexity2)
    p2=p2+perplexity2.item()
    cos=cos+calculate_sbert_similarity([generated_text,text],'/home/buhaoran2023/LLM_Models/sentence-transformers/all-MiniLM-L6-v2').item()
    print("cos:",cos)
print("p1:{},p2:{},cos:{}".format(p1,p2,cos/test_len))