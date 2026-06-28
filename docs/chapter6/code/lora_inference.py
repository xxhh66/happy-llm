'''
LoRA 模型推理脚本
用于加载基础模型 + LoRA 适配器进行推理
'''

import torch
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from typing import Optional, List, Dict
import logging

# ==================== 配置区 ====================
path = r"D:/workspace/deeplearning/d2cv/happy-llm/docs/"
BASE_MODEL_PATH = path + "autodl-tmp/qwen2.5-0.5B"  # 基础模型路径
LORA_ADAPTER_PATH = path + "autodl-tmp/lora_sft"    # LoRA 适配器路径

# 推理参数
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 50
DO_SAMPLE = True
REPETITION_PENALTY = 1.1

# 系统提示词
SYSTEM_MESSAGE = "You are a helpful assistant."
# ==============================================

logger = logging.getLogger(__name__)


class LoRAInference:
    """
    LoRA 模型推理类
    """
    
    def __init__(
        self, 
        base_model_path: str, 
        lora_adapter_path: str, 
        device: str = "auto",
        merge_weights: bool = False,
    ):
        """
        初始化推理模型
        
        Args:
            base_model_path: 基础模型路径
            lora_adapter_path: LoRA 适配器路径
            device: 设备
            merge_weights: 是否合并 LoRA 权重（合并后推理更快）
        """
        self.device = device
        self.merge_weights = merge_weights
        
        # 加载 Tokenizer
        logger.info(f"加载 Tokenizer: {base_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, 
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载基础模型
        logger.info(f"加载基础模型: {base_model_path}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto" if device == "auto" else device,
            low_cpu_mem_usage=True,
        )
        
        # 加载 LoRA 适配器
        logger.info(f"加载 LoRA 适配器: {lora_adapter_path}")
        self.model = PeftModel.from_pretrained(
            self.base_model, 
            lora_adapter_path,
            device_map="auto" if device == "auto" else device,
        )
        
        # 如果需要合并权重（加速推理）
        if merge_weights:
            logger.info("合并 LoRA 权重...")
            self.model = self.model.merge_and_unload()
            logger.info("✅ 权重合并完成")
        
        self.model.eval()
        logger.info("✅ LoRA 模型加载完成")
    
    def build_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """
        构建对话模板
        """
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": SYSTEM_MESSAGE}] + messages
        
        text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                text += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                text += f"<|im_start|>human\n{content}<|im_end|>\n"
            elif role == "assistant":
                text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            else:
                text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        
        text += "<|im_start|>assistant\n"
        return text
    
    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
        top_k: int = TOP_K,
        do_sample: bool = DO_SAMPLE,
        repetition_penalty: float = REPETITION_PENALTY,
    ) -> str:
        """
        生成回复
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        text = self.build_chat_template(messages)
        
        inputs = self.tokenizer(
            text, 
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return response.strip()
    
    def chat(self, prompt: str, **kwargs) -> str:
        """简化的对话接口"""
        return self.generate(prompt, **kwargs)
    
    def interactive_chat(self):
        """
        交互式对话模式
        """
        print("\n" + "=" * 60)
        print("LoRA 模型交互对话")
        print(f"基础模型: {BASE_MODEL_PATH}")
        print(f"LoRA适配器: {LORA_ADAPTER_PATH}")
        print("输入 'quit' 或 'exit' 退出对话")
        print("输入 'clear' 清空对话历史")
        print("=" * 60)
        
        history = []
        
        while True:
            try:
                user_input = input("\n👤 用户: ")
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 再见！")
                    break
                
                if user_input.lower() == 'clear':
                    history = []
                    print("🧹 对话历史已清空")
                    continue
                
                if not user_input.strip():
                    continue
                
                messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
                messages.extend(history)
                messages.append({"role": "user", "content": user_input})
                
                text = self.build_chat_template(messages)
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        do_sample=DO_SAMPLE,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                
                generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
                response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                response = response.strip()
                
                print(f"🤖 助手: {response}")
                
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": response})
                
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 错误: {e}")
                import traceback
                traceback.print_exc()
    
    def get_trainable_parameters_info(self):
        """
        获取 LoRA 可训练参数信息
        """
        trainable_params = 0
        all_params = 0
        for name, param in self.model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        print(f"\n📊 LoRA 模型参数统计:")
        print(f"  可训练参数: {trainable_params / 1e6:.2f}M")
        print(f"  总参数: {all_params / 1e6:.2f}M")
        print(f"  可训练占比: {100 * trainable_params / all_params:.2f}%")
        
        return trainable_params, all_params


def main():
    """主函数"""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # 检查路径是否存在
    if not os.path.exists(BASE_MODEL_PATH):
        print(f"⚠️ 基础模型不存在: {BASE_MODEL_PATH}")
        return
    
    if not os.path.exists(LORA_ADAPTER_PATH):
        print(f"⚠️ LoRA 适配器不存在: {LORA_ADAPTER_PATH}")
        print("请先运行 LoRA 训练脚本生成适配器")
        return
    
    # 初始化推理
    print("正在加载 LoRA 模型...")
    inference = LoRAInference(
        base_model_path=BASE_MODEL_PATH,
        lora_adapter_path=LORA_ADAPTER_PATH,
        merge_weights=False,  # 设为 True 可以加速推理
    )
    
    # 显示模型信息
    inference.get_trainable_parameters_info()
    
    # 单次测试
    print("\n" + "=" * 60)
    print("单次测试")
    print("=" * 60)
    test_prompt = "什么是机器学习？请简要说明。"
    print(f"👤 用户: {test_prompt}")
    response = inference.chat(test_prompt)
    print(f"🤖 助手: {response}")
    
    # 交互式对话
    inference.interactive_chat()


if __name__ == "__main__":
    main()