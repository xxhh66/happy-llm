import os

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 下载模型
# os.system('huggingface-cli download --resume-download Qwen/Qwen2.5-1.5B --local-dir autodl-tmp/qwen-1.5b')
os.system(r'hf download --repo-type model --force-download Qwen/Qwen2.5-0.5B --local-dir D:/workspace/deeplearning/d2cv/happy-llm/docs/autodl-tmp/qwen-0.5b')
