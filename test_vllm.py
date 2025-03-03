from vllm import LLM

# 定义要使用的 GPU 设备，这里指定为 GPU 4
device = "cuda:4"

# 初始化 LLM 并指定模型的路径和设备
llm = LLM(model="/home/buhaoran2023/LLM_Models/Meta/Llama2/Llama-2-7b-hf", device=device)

# 进行推理，如生成文本
prompts = ["请生成一段描述美丽风景的话", "请为一个科技产品写一段宣传文案"]
outputs = llm.generate(prompts)

# 输出结果
for output in outputs:
    print(output.prompt)
    print(output.outputs[0].text)