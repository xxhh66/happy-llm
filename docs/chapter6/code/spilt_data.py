import json

def detect_file_format(input_file):
    """
    检测文件的实际格式
    """
    print(f"检测文件格式: {input_file}")
    print("=" * 60)
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            # 读取前几个字符
            first_chars = f.read(100)
            print(f"文件开头: {repr(first_chars[:50])}")
            
            # 判断格式
            if first_chars.strip().startswith('['):
                print("✅ 格式: JSON数组 (以 '[' 开头)")
                return 'json_array'
            elif first_chars.strip().startswith('{'):
                print("✅ 格式: JSONL (每行一个JSON对象，以 '{' 开头)")
                return 'jsonl'
            elif first_chars.strip().startswith('['):
                print("✅ 格式: JSON数组")
                return 'json_array'
            else:
                print(f"⚠️ 未知格式，开头字符: {repr(first_chars[:20])}")
                return 'unknown'
                
    except Exception as e:
        print(f"❌ 检测失败: {e}")
        return 'error'

def extract_first_n_jsonl(input_file, output_file, n=100):
    """
    从JSONL格式文件中提取前N条记录
    """
    try:
        print(f"开始从JSONL文件提取前 {n} 条记录...")
        
        mini_data = []
        count = 0
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        mini_data.append(obj)
                        count += 1
                        
                        if count % 100 == 0:
                            print(f"已提取 {count} 条记录...")
                        
                        if count >= n:
                            # 保存为JSONL格式
                            with open(output_file, 'w', encoding='utf-8') as out:
                                for item in mini_data:
                                    out.write(json.dumps(item, ensure_ascii=False) + '\n')
                            print(f"\n✅ 成功提取 {count} 条记录")
                            print(f"📁 保存到: {output_file}")
                            print(f"📊 文件格式: JSONL (每行一个JSON对象)")
                            return mini_data
                    except json.JSONDecodeError as e:
                        print(f"⚠️ 跳过无效行: {e}")
                        continue
        
        # 如果没达到n条，保存所有
        if mini_data:
            with open(output_file, 'w', encoding='utf-8') as out:
                for item in mini_data:
                    out.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"\n✅ 成功提取 {count} 条记录")
            print(f"📁 保存到: {output_file}")
            return mini_data
        else:
            print("❌ 未提取到任何记录")
            return None
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_first_n_streaming_robust(input_file, output_file, n=100):
    """
    稳健的流式提取，自动检测并适配文件格式
    """
    try:
        # 先检测格式
        file_format = detect_file_format(input_file)
        
        if file_format == 'jsonl':
            print("\n📌 使用JSONL提取模式")
            return extract_first_n_jsonl(input_file, output_file, n)
        elif file_format == 'json_array':
            print("\n📌 使用JSON数组提取模式")
            # 这里可以用之前的流式提取函数
            return extract_first_n_streaming(input_file, output_file, n)
        else:
            # 尝试作为JSONL处理（最常见的格式）
            print("\n📌 尝试作为JSONL处理...")
            return extract_first_n_jsonl(input_file, output_file, n)
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return None

# 原来的流式提取函数（用于JSON数组）
def extract_first_n_streaming(input_file, output_file, n=100):
    """
    流式提取前N条记录（适用于JSON数组格式）
    """
    try:
        print(f"开始从JSON数组提取前 {n} 条记录...")
        
        mini_data = []
        count = 0
        current_obj = ""
        brace_depth = 0
        in_string = False
        in_object = False
        
        with open(input_file, 'r', encoding='utf-8') as f:
            # 跳过开头的 '[' 和空白
            char = f.read(1)
            while char and char.isspace():
                char = f.read(1)
            
            if char != '[':
                raise ValueError("JSON格式不是以 '[' 开头")
            
            # 逐字符读取
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                
                for char in chunk:
                    # 处理转义字符
                    if char == '\\' and (len(current_obj) == 0 or current_obj[-1] != '\\'):
                        current_obj += char
                        continue
                    
                    # 处理字符串引号
                    if char == '"' and (len(current_obj) == 0 or current_obj[-1] != '\\'):
                        in_string = not in_string
                    
                    # 不在字符串内时跟踪大括号深度
                    if not in_string:
                        if char == '{':
                            brace_depth += 1
                            in_object = True
                        elif char == '}':
                            brace_depth -= 1
                    
                    current_obj += char
                    
                    # 当一个完整的对象结束时
                    if in_object and brace_depth == 0 and not in_string:
                        if char in [',', ']']:
                            # 提取完整的JSON对象
                            obj_str = current_obj.strip().rstrip(',').strip()
                            if obj_str:
                                try:
                                    obj = json.loads(obj_str)
                                    mini_data.append(obj)
                                    count += 1
                                    
                                    if count % 100 == 0:
                                        print(f"已提取 {count} 条记录...")
                                    
                                    if count >= n:
                                        # 保存为JSONL格式
                                        with open(output_file, 'w', encoding='utf-8') as out:
                                            for item in mini_data:
                                                out.write(json.dumps(item, ensure_ascii=False) + '\n')
                                        print(f"\n✅ 成功提取 {count} 条记录")
                                        print(f"📁 保存到: {output_file}")
                                        print(f"📊 文件格式: JSONL (每行一个JSON对象)")
                                        return mini_data
                                        
                                except json.JSONDecodeError:
                                    pass
                            
                            current_obj = ""
                            in_object = False
        
        # 如果没达到n条，保存所有
        if mini_data:
            with open(output_file, 'w', encoding='utf-8') as out:
                for item in mini_data:
                    out.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"\n✅ 成功提取 {count} 条记录")
            print(f"📁 保存到: {output_file}")
            return mini_data
        else:
            print("❌ 未提取到任何记录")
            return None
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return None

# 使用示例
if __name__ == "__main__":
    path = r"D:/workspace/deeplearning/d2cv/happy-llm/docs/"
    
    # 先检测文件格式
    print("=" * 60)
    print("步骤1: 检测文件格式")
    print("=" * 60)
    detect_file_format(
        path + "dataset/BelleGroup/train_3.5M_CN.json"
    )
    
    print("\n" + "=" * 60)
    print("步骤2: 提取mini数据集")
    print("=" * 60)
    
    # 使用稳健的提取函数
    extract_first_n_streaming_robust(
        input_file=path + "dataset/BelleGroup/train_3.5M_CN.json",
        output_file=path + "dataset/BelleGroup/BelleGroup_mini_5000.jsonl",
        n=5000
    )