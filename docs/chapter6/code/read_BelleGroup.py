import json

# 如果你有完整的 BelleGroup 数据文件
file_path = r"D:/workspace/deeplearning/d2cv/happy-llm/docs/dataset/BelleGroup/train_3.5M_CN.json"
# 数据集格式 对话
# {
#   "conversations": [
#     {"from": "human", "value": "用户的问题"},
#     {"from": "assistant", "value": "AI的回答"}
#   ]
# }
# 读取前 3 条数据
with open(file_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        data = json.loads(line)
        print(f"样本 {i+1}:")
        print(json.dumps(data, ensure_ascii=False, indent=2))
        print("-" * 60)