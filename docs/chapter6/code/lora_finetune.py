'''
LoRA 微调脚本
基于 peft 库实现高效微调
原理：通过低秩分解（Low-Rank Adaptation）在冻结预训练模型的基础上，
只训练少量参数（LoRA层），大幅降低显存占用和训练成本
'''

import logging
import os
import sys
import json
from dataclasses import dataclass, field
from typing import Optional, Dict
from torch.utils.data import Dataset
import torch
from tqdm import tqdm

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
from transformers.trainer_utils import get_last_checkpoint

# PEFT 相关：Parameter-Efficient Fine-Tuning 参数高效微调库
from peft import (
    LoraConfig,           # LoRA 配置类
    get_peft_model,       # 将基础模型转换为 PEFT 模型
    TaskType,             # 任务类型（如 CAUSAL_LM 因果语言模型）
    PeftModel,            # PEFT 模型类
    PeftConfig,           # PEFT 配置类
    prepare_model_for_kbit_training,  # 准备模型进行量化训练
)

import swanlab  # 训练监控工具，类似 wandb

# ==================== 配置区 ====================
path = r"D:/workspace/deeplearning/d2cv/happy-llm/docs/"
MODEL_NAME_OR_PATH = path + "autodl-tmp/qwen2.5-0.5B"  # 基础模型路径
TRAIN_DATA_PATH = path + "dataset/BelleGroup/BelleGroup_5000.jsonl"  # 训练数据路径
OUTPUT_DIR = path + "autodl-tmp/lora_sft"  # 输出目录

# LoRA 超参数
LORA_R = 8  # LoRA 秩（rank），控制低秩矩阵的维度。值越大，可训练参数越多，表达能力越强
LORA_ALPHA = 32  # 缩放参数，用于控制 LoRA 权重的影响程度。通常 alpha = 2 * r
LORA_DROPOUT = 0.1  # Dropout 比例，防止过拟合
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]  # 要应用 LoRA 的模块
# q_proj: Query 投影层
# k_proj: Key 投影层  
# v_proj: Value 投影层
# o_proj: Output 投影层（注意力输出）

# 训练超参数
PER_DEVICE_TRAIN_BATCH_SIZE = 4  # 每个设备的批次大小
GRADIENT_ACCUMULATION_STEPS = 4  # 梯度累积步数，有效批次大小 = 4 * 4 = 16
NUM_TRAIN_EPOCHS = 3  # 训练轮数
LEARNING_RATE = 2e-4  # 学习率（LoRA 通常使用比全量微调更大的学习率）
MAX_LEN = 512  # 最大序列长度
SAVE_STEPS = 50  # 每多少步保存一次模型
LOGGING_STEPS = 10  # 每多少步打印一次日志
WARMUP_RATIO = 0.1  # 预热步数占总步数的比例

# 其他配置
SEED = 42  # 随机种子，保证可重现性
REPORT_TO = "none"  # 日志上报目标，"none" 表示不上报
# ==============================================

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


def load_training_data(data_path):
    """
    加载JSONL格式的训练数据
    
    JSONL格式：每行一个JSON对象，便于流式处理大文件
    
    Args:
        data_path: 数据文件路径
        
    Returns:
        list: 包含所有数据样本的列表
    """
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):  # 逐行读取，节省内存
            line = line.strip()
            if line:  # 跳过空行
                try:
                    data.append(json.loads(line))  # 解析JSON行
                except json.JSONDecodeError as e:
                    logger.warning(f"跳过第 {line_num} 行: {e}")  # 记录解析失败的行
    return data


def preprocess(sources, tokenizer, max_len, system_message: str = "You are a helpful assistant."):
    """
    将原始对话数据转换为模型训练所需的格式
    
    处理流程：
    1. 构建对话模板：<|im_start|>system\n系统提示<|im_end|>\n
    2. 拼接多轮对话：human 和 assistant 交替
    3. 设置标签掩码：只对 assistant 的回答计算损失
    
    Args:
        sources: 原始对话数据列表
        tokenizer: 分词器
        max_len: 最大序列长度
        system_message: 系统提示词
        
    Returns:
        dict: 包含 input_ids, labels, attention_mask
    """
    # 角色映射：数据集中的角色名 -> Qwen 格式的角色标记
    roles = {"human": "<|im_start|>human", "assistant": "<|im_start|>assistant"}

    # 获取特殊标记的 token ID
    im_start = tokenizer("<|im_start|>").input_ids  # 消息开始标记
    im_end = tokenizer("<|im_end|>").input_ids      # 消息结束标记
    IGNORE_TOKEN_ID = tokenizer.pad_token_id        # 忽略标记ID（-100），该位置的损失不计算
    nl_tokens = tokenizer('\n').input_ids           # 换行符
    
    # 角色标识符：角色名 + 换行符
    _system = tokenizer('system').input_ids + nl_tokens      # system\n
    _user = tokenizer('human').input_ids + nl_tokens         # human\n
    _assistant = tokenizer('assistant').input_ids + nl_tokens # assistant\n

    input_ids, targets = [], []  # input_ids: 模型输入, targets: 训练标签

    # 遍历所有对话样本
    for i in tqdm(range(len(sources)), desc="预处理数据"):
        source = sources[i]
        
        # 兼容不同的数据格式,单轮对话，多轮对话等
        if isinstance(source, dict):
            if "conversations" in source:
                source = source["conversations"]
            elif "conversation" in source:
                source = source["conversation"]
            elif "dialog" in source:
                source = source["dialog"]
        
        if not isinstance(source, list):
            continue  # 跳过非列表格式的数据
            
        # 确保对话从 human 开始
        if source and source[0].get("from") != "human":
            for idx, turn in enumerate(source):
                if turn.get("from") == "human":
                    source = source[idx:]
                    break
        
        # 至少需要一轮 human-assistant 对话
        if not source or len(source) < 2:
            continue
            
        input_id, target = [], []  # 当前样本的输入和标签

        # 构建 System Prompt
        # 格式: <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
        system = im_start + _system + tokenizer(system_message).input_ids + im_end + nl_tokens
        input_id += system
        # System 部分不计算损失，全部标记为 IGNORE_TOKEN_ID
        target += im_start + [IGNORE_TOKEN_ID] * (len(system)-3) + im_end + nl_tokens
        assert len(input_id) == len(target)  # 确保长度一致

        # 处理每一轮对话
        for j, sentence in enumerate(source):
            if "from" not in sentence or "value" not in sentence:
                continue  # 跳过格式不完整的轮次
                
            role = roles.get(sentence["from"])  # 获取角色标记
            if role is None:
                continue  # 跳过未知角色
                
            # 构建当前轮次的输入
            # 格式: <|im_start|>role\ncontent<|im_end|>\n
            _input_id = tokenizer(role).input_ids + nl_tokens + \
                tokenizer(sentence["value"]).input_ids + im_end + nl_tokens
            input_id += _input_id

            # 构建当前轮次的标签（控制哪些位置计算损失）
            if role == '<|im_start|>human':
                # human 部分不计算损失
                _target = im_start + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + im_end + nl_tokens
            elif role == '<|im_start|>assistant':
                # assistant 部分计算损失（这是模型要学习的内容）
                _target = im_start + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                    _input_id[len(tokenizer(role).input_ids)+1:-2] + im_end + nl_tokens
            else:
                raise NotImplementedError(f"未知角色: {role}")
                
            target += _target

        assert len(input_id) == len(target)  # 确保长度一致

        # 填充（Padding）到固定长度
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        
        # 截取前 max_len 个 token
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])

    if not input_ids:
        raise ValueError("没有有效的训练样本！")  # 如果没有有效样本则报错

    # 转换为 PyTorch 张量
    input_ids = torch.tensor(input_ids)
    targets = torch.tensor(targets)

    return dict(
        input_ids=input_ids,           # 模型输入
        labels=targets,                # 训练目标（仅 assistant 部分有效）
        attention_mask=input_ids.ne(tokenizer.pad_token_id),  # 注意力掩码，忽略 pad 位置
    )


class SupervisedDataset(Dataset):
    """
    有监督微调数据集类
    继承自 torch.utils.data.Dataset，用于 Trainer 训练
    """
    def __init__(self, raw_data, tokenizer, max_len: int):
        super(SupervisedDataset, self).__init__()
        # 预处理数据
        data_dict = preprocess(raw_data, tokenizer, max_len)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]
        
        logger.info(f"✅ 数据集创建完成，共 {len(self)} 个有效样本")

    def __len__(self):
        """返回数据集大小"""
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        """获取第 i 个样本"""
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


def main():
    """
    主函数：执行 LoRA 微调全流程
    
    流程：
    1. 加载 Tokenizer
    2. 加载基础模型
    3. 配置 LoRA 参数
    4. 应用 LoRA 到模型
    5. 加载训练数据
    6. 创建数据集
    7. 配置训练参数
    8. 创建 Trainer
    9. 开始训练
    10. 保存模型
    """
    # ========== 设置日志 ==========
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    # ========== 设置随机种子 ==========
    set_seed(SEED)  # 保证训练结果可重现

    # ========== 1. 加载 Tokenizer ==========
    logger.info(f"加载 Tokenizer: {MODEL_NAME_OR_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, trust_remote_code=True)
    
    # 如果 tokenizer 没有 pad_token，使用 eos_token 代替
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ========== 2. 加载基础模型 ==========
    logger.info(f"加载模型: {MODEL_NAME_OR_PATH}")
    # 注意：不使用 device_map="auto"，让 Trainer 管理设备分配
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float16,  # 半精度，节省显存
        low_cpu_mem_usage=True,     # 节省 CPU 内存
        use_cache=False,            # 与 gradient_checkpointing 兼容
    )
    
    # ========== 3. 配置 LoRA ==========
    logger.info("配置 LoRA 参数...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # 因果语言模型任务
        inference_mode=False,           # 训练模式（非推理模式）
        r=LORA_R,                       # LoRA 秩
        lora_alpha=LORA_ALPHA,          # 缩放参数
        lora_dropout=LORA_DROPOUT,      # Dropout 比例
        target_modules=LORA_TARGET_MODULES,  # 目标模块
        bias="none",                    # 不训练偏置项
    )
    
    logger.info(f"LoRA 配置:")
    logger.info(f"  r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    logger.info(f"  target_modules={LORA_TARGET_MODULES}")

    # ========== 4. 准备模型进行训练 ==========
    logger.info("准备模型进行 LoRA 训练...")
    
    # 启用梯度检查点（用计算换显存，节省约 30% 显存）
    model.gradient_checkpointing_enable()
    
    # 应用 LoRA：将基础模型转换为 PEFT 模型
    # 这会冻结原参数，只保留 LoRA 层可训练
    model = get_peft_model(model, lora_config)
    
    # 打印可训练参数统计
    trainable_params = 0
    all_params = 0
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    logger.info(f"  可训练参数: {trainable_params / 1e6:.2f}M")
    logger.info(f"  总参数: {all_params / 1e6:.2f}M")
    logger.info(f"  可训练参数占比: {100 * trainable_params / all_params:.2f}%")
    
    model.print_trainable_parameters()  # peft 提供的便捷方法

    # ========== 5. 加载训练数据 ==========
    logger.info(f"加载训练数据: {TRAIN_DATA_PATH}")
    raw_data = load_training_data(TRAIN_DATA_PATH)
    logger.info(f"✅ 成功加载 {len(raw_data)} 条原始数据")

    # ========== 6. 创建数据集 ==========
    logger.info("创建训练数据集...")
    train_dataset = SupervisedDataset(raw_data, tokenizer=tokenizer, max_len=MAX_LEN)

    # ========== 7. 配置训练参数 ==========
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,          # 输出目录
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,  # 每设备批次大小
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,  # 梯度累积步数
        logging_steps=LOGGING_STEPS,    # 日志打印间隔
        num_train_epochs=NUM_TRAIN_EPOCHS,  # 训练轮数
        save_steps=SAVE_STEPS,          # 模型保存间隔
        learning_rate=LEARNING_RATE,    # 学习率
        warmup_ratio=WARMUP_RATIO,      # 预热比例
        report_to=REPORT_TO,            # 日志上报目标
        fp16=True,                      # 混合精度训练（16位浮点数）
        gradient_checkpointing=True,    # 梯度检查点（节省显存）
        save_total_limit=2,             # 最多保留 2 个 checkpoint
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),  # 日志目录
        overwrite_output_dir=True,      # 覆盖输出目录
        dataloader_num_workers=0,       # Windows 下避免多进程问题
        remove_unused_columns=False,    # 保留所有列
        no_cuda=False,                  # 使用 CUDA
    )

    logger.info(f"训练参数: {training_args}")

    # ========== 8. 创建 Trainer ==========
    # Trainer 封装了训练循环、分布式训练、日志记录等功能
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,  # 默认数据整理器
        # tokenizer 参数已被弃用，不再传递
    )

    # ========== 9. 开始训练 ==========
    logger.info("开始 LoRA 微调训练...")
    try:
        # 显式地将模型移动到指定设备
        model.to(training_args.device)
        
        # 执行训练
        train_result = trainer.train()
        
        # ========== 10. 保存模型 ==========
        # 保存 LoRA 适配器（只保存可训练参数，文件很小）
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        logger.info(f"✅ LoRA 模型已保存到: {OUTPUT_DIR}")
        
        # 保存训练指标
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
    except Exception as e:
        logger.error(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()  # 打印完整的错误堆栈


if __name__ == "__main__":
    main()  # 程序入口