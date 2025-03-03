# Copyright 2024 THU-BPM MarkLLM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ============================================
# kgw.py
# Description: Implementation of KGW algorithm
# ============================================

import torch  # 导入PyTorch库
from math import sqrt  # 导入平方根函数
from functools import partial   # 导入partial函数
from ..base import BaseWatermark  # 导入BaseWatermark基类
from utils.utils import load_config_file  # 导入加载配置文件的函数
from utils.transformers_config import TransformersConfig  # 导入TransformersConfig类
from exceptions.exceptions import AlgorithmNameMismatchError  # 导入算法名称不匹配异常
from transformers import LogitsProcessor, LogitsProcessorList  # 导入LogitsProcessor和LogitsProcessorList
from visualize.data_for_visualization import DataForVisualization  # 导入数据可视化类


class KGWConfig:
    """Config class for KGW algorithm, load config file and initialize parameters."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
            Initialize the KGW configuration.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        if algorithm_config is None:
            config_dict = load_config_file('config/KGW.json')  # 如果未提供配置文件路径，则加载默认配置文件
        else:
            config_dict = load_config_file(algorithm_config)  # 加载指定的配置文件
        if config_dict['algorithm_name'] != 'KGW':
            raise AlgorithmNameMismatchError('KGW', config_dict['algorithm_name'])  # 检查算法名称是否匹配

        self.gamma = config_dict['gamma']  # 加载gamma参数
        self.delta = config_dict['delta']  # 加载delta参数
        self.hash_key = config_dict['hash_key']  # 加载hash_key参数
        self.z_threshold = config_dict['z_threshold']  # 加载z_threshold参数
        self.prefix_length = config_dict['prefix_length']  # 加载prefix_length参数
        self.f_scheme = config_dict['f_scheme']  # 加载f_scheme参数
        self.window_scheme = config_dict['window_scheme']  # 加载window_scheme参数

        self.generation_model = transformers_config.model  # 加载生成模型
        self.generation_tokenizer = transformers_config.tokenizer  # 加载生成分词器
        self.vocab_size = transformers_config.vocab_size  # 加载词汇表大小
        self.device = transformers_config.device  # 加载设备（CPU或GPU）
        self.gen_kwargs = transformers_config.gen_kwargs  # 加载生成参数


class KGWUtils:
    """Utility class for KGW algorithm, contains helper functions."""

    def __init__(self, config: KGWConfig, *args, **kwargs) -> None:
        """
            Initialize the KGW utility class.

            Parameters:
                config (KGWConfig): Configuration for the KGW algorithm.
        """
        self.config = config  # 保存配置对象
        self.rng = torch.Generator(device=self.config.device)  # 创建随机数生成器
        self.rng.manual_seed(self.config.hash_key)  # 设置随机数生成器的种子
        self.prf = torch.randperm(self.config.vocab_size, device=self.config.device, generator=self.rng)  # 生成伪随机排列
        self.f_scheme_map = {"time": self._f_time, "additive": self._f_additive, "skip": self._f_skip,
                             "min": self._f_min}  # 映射前缀计算方案
        self.window_scheme_map = {"left": self._get_greenlist_ids_left, "self": self._get_greenlist_ids_self}  # 映射窗口方案

    def _f(self, input_ids: torch.LongTensor) -> int:
        """Get the previous token."""
        return int(self.f_scheme_map[self.config.f_scheme](input_ids))  # 根据配置选择前缀计算方案

    def _f_time(self, input_ids: torch.LongTensor):
        """Get the previous token time."""
        time_result = 1
        for i in range(0, self.config.prefix_length):
            time_result *= input_ids[-1 - i].item()  # 计算时间乘积
        return self.prf[time_result % self.config.vocab_size]  # 返回伪随机排列中的值

    def _f_additive(self, input_ids: torch.LongTensor):
        """Get the previous token additive."""
        additive_result = 0
        for i in range(0, self.config.prefix_length):
            additive_result += input_ids[-1 - i].item()  # 计算加法结果
        return self.prf[additive_result % self.config.vocab_size]  # 返回伪随机排列中的值

    def _f_skip(self, input_ids: torch.LongTensor):
        """Get the previous token skip."""
        return self.prf[input_ids[- self.config.prefix_length].item()]  # 返回指定位置的伪随机排列值

    def _f_min(self, input_ids: torch.LongTensor):
        """Get the previous token min."""
        return min(self.prf[input_ids[-1 - i].item()] for i in range(0, self.config.prefix_length))  # 返回最小值

    def get_greenlist_ids(self, input_ids: torch.LongTensor) -> list[int]:
        """Get greenlist ids for the input_ids."""
        return self.window_scheme_map[self.config.window_scheme](input_ids)  # 根据配置选择窗口方案

    def _get_greenlist_ids_left(self, input_ids: torch.LongTensor) -> list[int]:
        """Get greenlist ids for the input_ids via leftHash scheme."""
        self.rng.manual_seed((self.config.hash_key * self._f(input_ids)) % self.config.vocab_size)  # 设置随机数生成器的种子
        greenlist_size = int(self.config.vocab_size * self.config.gamma)  # 计算绿色列表大小
        vocab_permutation = torch.randperm(self.config.vocab_size, device=input_ids.device,
                                           generator=self.rng)  # 生成词汇表排列
        greenlist_ids = vocab_permutation[:greenlist_size]  # 获取绿色列表ID
        return greenlist_ids

    def _get_greenlist_ids_self(self, input_ids: torch.LongTensor) -> list[int]:
        """Get greenlist ids for the input_ids via selfHash scheme."""
        greenlist_size = int(self.config.vocab_size * self.config.gamma)  # 计算绿色列表大小
        greenlist_ids = []
        f_x = self._f(input_ids)  # 获取前缀token
        for k in range(0, self.config.vocab_size):
            h_k = f_x * int(self.prf[k])  # 计算哈希值
            self.rng.manual_seed(h_k % self.config.vocab_size)  # 设置随机数生成器的种子
            vocab_permutation = torch.randperm(self.config.vocab_size, device=input_ids.device,
                                               generator=self.rng)  # 生成词汇表排列
            temp_greenlist_ids = vocab_permutation[:greenlist_size]  # 获取临时绿色列表ID
            if k in temp_greenlist_ids:
                greenlist_ids.append(k)  # 将符合条件的ID加入绿色列表
        print(len(greenlist_ids),greenlist_size)
        return greenlist_ids

    def _compute_z_score(self, observed_count: int, T: int) -> float:
        """Compute z-score for the given observed count and total tokens."""
        expected_count = self.config.gamma  # 获取期望计数
        numer = observed_count - expected_count * T  # 计算分子
        denom = sqrt(T * expected_count * (1 - expected_count))  # 计算分母
        z = numer / denom  # 计算z分数
        return z

    def score_sequence(self, input_ids: torch.Tensor) -> tuple[float, list[int]]:
        """Score the input_ids and return z_score and green_token_flags."""
        num_tokens_scored = len(input_ids) - self.config.prefix_length  # 计算可评分的token数量
        if num_tokens_scored < 1:
            raise ValueError(
                (
                    f"Must have at least {1} token to score after "
                    f"the first min_prefix_len={self.config.prefix_length} tokens required by the seeding scheme."
                )
            )

        green_token_count = 0  # 初始化绿色token计数
        green_token_flags = [-1 for _ in range(self.config.prefix_length)]  # 初始化绿色token标志

        for idx in range(self.config.prefix_length, len(input_ids)):
            curr_token = input_ids[idx]  # 获取当前token
            greenlist_ids = self.get_greenlist_ids(input_ids[:idx])  # 获取绿色列表ID
            if curr_token in greenlist_ids:
                green_token_count += 1  # 增加绿色token计数
                green_token_flags.append(1)  # 设置绿色token标志
            else:
                green_token_flags.append(0)  # 设置非绿色token标志

        z_score = self._compute_z_score(green_token_count, num_tokens_scored)  # 计算z分数
        return z_score, green_token_flags  # 返回z分数和绿色token标志


class KGWLogitsProcessor(LogitsProcessor):
    """LogitsProcessor for KGW algorithm, process logits to add watermark."""

    def __init__(self, config: KGWConfig, utils: KGWUtils, *args, **kwargs) -> None:
        """
            Initialize the KGW logits processor.

            Parameters:
                config (KGWConfig): Configuration for the KGW algorithm.
                utils (KGWUtils): Utility class for the KGW algorithm.
        """
        self.config = config  # 保存配置对象
        self.utils = utils  # 保存工具类对象

    def _calc_greenlist_mask(self, scores: torch.FloatTensor,
                             greenlist_token_ids: torch.LongTensor) -> torch.BoolTensor:
        """Calculate greenlist mask for the given scores and greenlist token ids."""
        green_tokens_mask = torch.zeros_like(scores)  # 初始化掩码张量，与scores形状相同
        for b_idx in range(len(greenlist_token_ids)):  # 遍历每个batch
            green_tokens_mask[b_idx][greenlist_token_ids[b_idx]] = 1  # 设置绿色列表token的位置为1
        final_mask = green_tokens_mask.bool()  # 将掩码转换为布尔类型
        return final_mask  # 返回布尔掩码

    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor,
                               greenlist_bias: float) -> torch.Tensor:
        """Bias the scores for the greenlist tokens."""
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias  # 对绿色列表token的logits进行偏置
        return scores  # 返回偏置后的logits

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process logits to add watermark."""
        if input_ids.shape[-1] < self.config.prefix_length:
            return scores  # 如果输入序列长度小于前缀长度，直接返回原始logits

        batched_greenlist_ids = [None for _ in range(input_ids.shape[0])]  # 初始化绿色列表ID列表

        for b_idx in range(input_ids.shape[0]):  # 遍历每个batch
            greenlist_ids = self.utils.get_greenlist_ids(input_ids[b_idx])  # 获取当前batch的绿色列表ID
            batched_greenlist_ids[b_idx] = greenlist_ids  # 保存绿色列表ID

        green_tokens_mask = self._calc_greenlist_mask(scores=scores,
                                                      greenlist_token_ids=batched_greenlist_ids)  # 计算绿色列表掩码

        scores = self._bias_greenlist_logits(scores=scores, greenlist_mask=green_tokens_mask,
                                             greenlist_bias=self.config.delta)  # 偏置绿色列表logits

        return scores  # 返回处理后的logits


class KGW(BaseWatermark):
    """Top-level class for KGW algorithm."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
            Initialize the KGW algorithm.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        self.config = KGWConfig(algorithm_config, transformers_config)  # 初始化配置对象
        self.utils = KGWUtils(self.config)  # 初始化工具类对象
        self.logits_processor = KGWLogitsProcessor(self.config, self.utils)  # 初始化logits处理器

    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """Generate watermarked text."""

        # Configure generate_with_watermark
        generate_with_watermark = partial(
            self.config.generation_model.generate,
            logits_processor=LogitsProcessorList([self.logits_processor]),
            **self.config.gen_kwargs
        )

        # Encode prompt
        encoded_prompt = self.config.generation_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(
            self.config.device)  # 编码提示文本
        # Generate watermarked text
        encoded_watermarked_text = generate_with_watermark(**encoded_prompt)  # 生成带水印的文本
        # Decode
        watermarked_text = \
        self.config.generation_tokenizer.batch_decode(encoded_watermarked_text, skip_special_tokens=True)[0]  # 解码生成的文本
        return watermarked_text  # 返回带水印的文本

    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs):
        """Detect watermark in the text."""

        # Encode the text
        encoded_text = \
        self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(
            self.config.device)  # 编码输入文本

        # Compute z_score using a utility method
        z_score, _ = self.utils.score_sequence(encoded_text)  # 计算z分数

        # Determine if the z_score indicates a watermark
        is_watermarked = z_score > self.config.z_threshold  # 判断是否包含水印

        # Return results based on the return_dict flag
        if return_dict:
            return {"is_watermarked": is_watermarked, "score": z_score}  # 返回字典格式的结果
        else:
            return (is_watermarked, z_score)  # 返回元组格式的结果

    def get_data_for_visualization(self, text: str, *args, **kwargs) -> tuple[list[str], list[int]]:
        """Get data for visualization."""

        # Encode text
        encoded_text = \
        self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(
            self.config.device)  # 编码输入文本

        # Compute z-score and highlight values
        z_score, highlight_values = self.utils.score_sequence(encoded_text)  # 计算z分数和高亮值

        # decode single tokens
        decoded_tokens = []
        for token_id in encoded_text:
            token = self.config.generation_tokenizer.decode(token_id.item())  # 解码单个token
            decoded_tokens.append(token)  # 保存解码后的token

        return DataForVisualization(decoded_tokens, highlight_values)  # 返回可视化数据
