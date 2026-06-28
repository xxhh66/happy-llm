'''
SFT 脚本
'''

import logging
import math
import os
import sys
from dataclasses import dataclass, field
# from torchdata.datapipes.iter import IterableWrapper
from torchdata.nodes import IterableWrapper
from itertools import chain
# import deepspeed
from typing import Optional,List,Dict
from torch.utils.data import Dataset
import json


import datasets
import pandas as pd
import torch
from datasets import load_dataset
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
import datetime
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
import swanlab
from tqdm import tqdm

path = r"D:/workspace/deeplearning/d2cv/happy-llm/docs/"
# ==================== 配置区 ====================
# 模型配置
MODEL_NAME_OR_PATH = path+"autodl-tmp/qwen2.5-0.5B"  # 本地模型路径，或 HuggingFace ID: "Qwen/Qwen2.5-0.5B"
TRAIN_DATA_PATH =path+ "dataset/BelleGroup/BelleGroup_5000.jsonl"  # 训练数据路径
OUTPUT_DIR =path+ "autodl-tmp/sft"  # 输出目录

# 训练超参数
PER_DEVICE_TRAIN_BATCH_SIZE = 4  # 每设备 batch size（4060 建议 1）
GRADIENT_ACCUMULATION_STEPS = 8  # 梯度累积步数
NUM_TRAIN_EPOCHS = 10  # 训练轮数
LEARNING_RATE = 1e-4  # 学习率
MAX_LEN = 512  # 最大序列长度
SAVE_STEPS = 100  # 保存间隔
LOGGING_STEPS = 10  # 日志间隔
WARMUP_STEPS = 200  # 预热步数

# 其他配置
SEED = 42
REPORT_TO = "none"  # 是否使用 wandb/swanlab
# ==============================================

logger = logging.getLogger(__name__)


# 超参类
@dataclass
class ModelArguments:
    """
    关于模型的参数
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "预训练模型参数地址"
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "模型训练使用的数据类型，推荐 bfloat16"
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )


@dataclass
class DataTrainingArguments:
    """
    关于训练的参数
    """

    train_files: Optional[str]  = field(default=None, metadata={"help": "训练数据路径"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "最大文本块长度"
            )
        },
    )

# 指令文本处理
# 参考：https://github.com/QwenLM/Qwen/blob/main/finetune.py
def preprocess(sources, tokenizer, max_len, system_message: str = "You are a helpful assistant."):
    """
    将原始对话数据转换为模型训练所需的格式
    
    Args:
        sources: 原始对话数据列表，每个元素包含 "conversations" 字段
        tokenizer: 分词器
        max_len: 最大序列长度
        system_message: 系统提示词，默认为 "You are a helpful assistant."
    
    Returns:
        dict: 包含 input_ids（输入）、labels（标签）、attention_mask（注意力掩码）
    """
    # ========== 1. 定义角色映射 ==========
    # 将数据集中的角色名映射为 Qwen 格式的角色标记
    roles = {"human": "<|im_start|>human", "assistant": "<|im_start|>assistant"}

    # ========== 2. 预定义特殊标记的 token ID ==========
    # BOS (Begin of Sentence) 对话开始标记
    # <|im_start|> 表示一段新消息的开始
    im_start = tokenizer("<|im_start|>").input_ids
    
    # EOS (End of Sentence) 对话结束标记
    # <|im_end|> 表示一段消息的结束
    im_end = tokenizer("<|im_end|>").input_ids
    
    # PAD 忽略标记 ID
    # 在计算损失时，标记为 IGNORE_TOKEN_ID 的位置会被跳过
    # 这样模型就不会从 system 和 user 的文本中学习
    IGNORE_TOKEN_ID = tokenizer.pad_token_id
    
    # 换行符的 token ID，用于格式化
    nl_tokens = tokenizer('\n').input_ids
    
    # 角色标识符（角色名 + 换行符）
    # 用于构建 system/human/assistant 的标记头
    _system = tokenizer('system').input_ids + nl_tokens      # system\n
    _user = tokenizer('human').input_ids + nl_tokens         # human\n
    _assistant = tokenizer('assistant').input_ids + nl_tokens # assistant\n

    # ========== 3. 初始化结果容器 ==========
    input_ids, targets = [], []  # input_ids: 模型输入, targets: 训练目标(labels)

    # ========== 4. 遍历所有对话样本 ==========
    for i in tqdm(range(len(sources))):
        source = sources[i]  # 当前对话样本
        
        # 确保对话从 human 开始（如果第一条不是 human，则跳过）
        if source[0]["from"] != "human":
            source = source[1:]
            
        # 当前样本的输入和目标序列
        input_id, target = [], []

        # ========== 5. 构建 System Prompt ==========
        # 格式: <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
        system = im_start + _system + tokenizer(system_message).input_ids + im_end + nl_tokens
        input_id += system
        
        # System 部分不需要计算损失，全部标记为 IGNORE_TOKEN_ID
        # len(system)-3 是因为 im_start 长度为 1，前后各有一个标记需要保留格式
        target += im_start + [IGNORE_TOKEN_ID] * (len(system)-3) + im_end + nl_tokens
        
        # 断言：输入和目标长度必须一致
        assert len(input_id) == len(target)

        # ========== 6. 遍历当前对话的每一轮 ==========
        for j, sentence in enumerate(source):
            # 获取当前句子的角色（human 或 assistant）
            role = roles[sentence["from"]]
            
            # ========== 7. 构建当前轮次的输入 ==========
            # User 格式: <|im_start|>human\n用户问题<|im_end|>\n
            # Assistant 格式: <|im_start|>assistant\nAI回答<|im_end|>\n
            _input_id = tokenizer(role).input_ids + nl_tokens + \
                tokenizer(sentence["value"]).input_ids + im_end + nl_tokens
            input_id += _input_id

            # ========== 8. 构建当前轮次的标签（控制学习目标） ==========
            if role == '<|im_start|>human':
                # ⚠️ User 部分不计算损失
                # 只保留格式标记，内容全部替换为 IGNORE_TOKEN_ID
                _target = im_start + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + im_end + nl_tokens
                
            elif role == '<|im_start|>assistant':
                # ✅ Assistant 部分需要计算损失（这是模型要学习的核心）
                # 保留 "assistant\n" 和实际回答内容，但将 "<|im_start|>" 替换为 IGNORE_TOKEN_ID
                _target = im_start + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                    _input_id[len(tokenizer(role).input_ids)+1:-2] + im_end + nl_tokens
            else:
                # 如果出现未知角色，报错
                print(role)
                raise NotImplementedError
                
            target += _target

        # 确保输入和目标长度一致
        assert len(input_id) == len(target)

        # ========== 9. 填充到固定长度（Padding） ==========
        # 如果当前序列长度小于 max_len，用 pad_token_id 填充输入
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        # 目标序列用 IGNORE_TOKEN_ID 填充（这些位置不计算损失）
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        
        # 截取前 max_len 个 token
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])

    # ========== 10. 转换为 PyTorch 张量 ==========
    input_ids = torch.tensor(input_ids)
    targets = torch.tensor(targets)

    # ========== 11. 返回训练所需的数据 ==========
    return dict(
        input_ids=input_ids,           # 模型输入
        labels=targets,                 # 训练目标（仅 assistant 部分有效）
        attention_mask=input_ids.ne(tokenizer.pad_token_id),  # 注意力掩码，忽略 pad 位置
    )
# 自定义一个 Dataset
from typing import Dict

class SupervisedDataset(Dataset):

    def __init__(self, raw_data, tokenizer, max_len: int):
        super(SupervisedDataset, self).__init__()
        # 加载并预处理数据
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, max_len)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )

                
def main():
    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    # 设置随机种子
    set_seed(SEED)

    # 1. 加载 Tokenizer
    logger.info(f"加载 Tokenizer: {MODEL_NAME_OR_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, trust_remote_code=True)

    # 2. 加载模型
    logger.info(f"加载模型: {MODEL_NAME_OR_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH,
        trust_remote_code=True,
        device_map="auto",           # 自动分配设备
        torch_dtype=torch.float16,   # 半精度，节省显存
        low_cpu_mem_usage=True,      # 节省 CPU 内存
    )

    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(f"模型参数量: {n_params / 2**20:.2f}M")

    # 3. 加载训练数据
    logger.info(f"加载训练数据: {TRAIN_DATA_PATH}")
    with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
        lst = [json.loads(line) for line in f.readlines()]

    logger.info(f"训练样本总数: {len(lst)}")

    # 4. 创建数据集
    train_dataset = SupervisedDataset(lst, tokenizer=tokenizer, max_len=MAX_LEN)

    # 5. 配置训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        logging_steps=LOGGING_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        save_steps=SAVE_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        report_to=REPORT_TO,
        bf16=True,                     # 混合精度训练
        # fp16=True,                     # 混合精度训练
        gradient_checkpointing=True,   # 节省显存
        save_total_limit=2,            # 只保留最近 2 个 checkpoint
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        overwrite_output_dir=True,     # 覆盖输出目录
        dataloader_num_workers=0,      # Windows 下避免多进程问题
    )

    logger.info(f"训练参数: {training_args}")

    # 6. 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    # 7. 开始训练
    logger.info("开始训练...")
    train_result = trainer.train()

    # 8. 保存模型
    trainer.save_model()
    logger.info(f"模型已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
