from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset
import torch
from itertools import chain
from transformers import TrainingArguments
from transformers import Trainer, default_data_collator

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 在加载模型前清理显存
torch.cuda.empty_cache()

block_size = 2048

# 全局变量和函数定义可以放在外面
model_path = r"D:/workspace/deeplearning/d2cv/happy-llm/docs/autodl-tmp/qwen2.5-0.5B"
ds_path = r"D:/workspace/deeplearning/d2cv/happy-llm/docs/dataset/pretrain_mini_500.jsonl"
output_dir=r"D:/workspace/deeplearning/d2cv/happy-llm/docs/autodl-tmp/output/pretrain"
def tokenize_function(examples, tokenizer):
    """分词函数（可放在外面）"""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=256,
        padding=False,
        return_tensors=None,
    )

def group_texts(examples):
    # 将文本段拼接起来
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    # 计算拼起来的整体长度
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # 如果长度太长，进行分块
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    # print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))       
    print("group texts input examples length%d after_group size%d"%(len(examples['input_ids']),len(result["input_ids"])))
    result["labels"] = result["input_ids"].copy()
    return result

# 主要执行代码放入 main 保护块
if __name__ == '__main__':
    # 1. 加载数据集
    ds = load_dataset('json', data_files=ds_path)
    print("=" * 60)
    print(f"数据集样本:\n{ds['train'][0]}")
    print("=" * 60)
    
    column_names = list(ds["train"].features)
    print(f"数据集特征列: {column_names}")
    print("=" * 60)

    # 2. 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_cache=False,
    )
    
    # 3. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 4. 使用闭包或 functools.partial 传递 tokenizer
    from functools import partial
    tokenize_with_tokenizer = partial(tokenize_function, tokenizer=tokenizer)

    # 5. 批量处理数据集（降低 num_proc 避免问题）
    tokenized_datasets = ds.map(
        tokenize_with_tokenizer,
        batched=True,
        batch_size=64,
        num_proc=1,  # 先改为 1，确认程序无误后再尝试调大
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    print("分词后的数据集:", tokenized_datasets)
    print("=" * 60)
    print(f"tokenized 后的数据列: {list(tokenized_datasets['train'].features.keys())}")
    # 6. 预训练一般将文本拼接成固定长度的文本段
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=1,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
        batch_size = 64,
    )
    train_dataset = lm_datasets["train"]
    print("="*60)
    print(train_dataset)
    print("="*60)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        logging_steps=10,
        num_train_epochs=10,
        save_steps=100,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="none",
    )
    # 如果 train_dataset 已经是一个 PyTorch Dataset 对象，直接传入即可
    # 无需任何包装
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,   # 直接传入你的 dataset 对象，移除 IterableWrapper
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=default_data_collator
    )
    print('start train')
    train_result = trainer.train()