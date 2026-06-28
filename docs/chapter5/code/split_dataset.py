import json
import os

def split_pretrain_dataset(input_file, output_file, num_lines=5000):
    """
    切分预训练数据集
    
    Args:
        input_file: 原始文件路径
        output_file: 输出文件路径
        num_lines: 保留的行数
    """
    print(f"正在处理: {input_file}")
    
    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 文件 {input_file} 不存在")
        return
    # 版本二：直接截取前N行，不进行所有行的遍历
    lines_written = 0
    with open(input_file, 'r', encoding='utf-8') as f_in:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for i, line in enumerate(f_in):
                # 当前行大于设定行数时，就break
                if i >= num_lines:
                    break
                f_out.write(line)
                lines_written += 1

    print(f"已保存 {lines_written:,} 行到: {output_file}")
    # # 版本一：获取文件总行数，耗时比较大
    # with open(input_file, 'r', encoding='utf-8') as f:
    #     total_lines = sum(1 for _ in f)
    # print(f"原始文件总行数: {total_lines:,}")
    
    # # 确定实际要保留的行数
    # num_lines = min(num_lines, total_lines)
    
    # # 读取并写入前 N 行
    # with open(input_file, 'r', encoding='utf-8') as f_in:
    #     with open(output_file, 'w', encoding='utf-8') as f_out:
    #         for i in range(num_lines):
    #             line = f_in.readline()
    #             if not line:
    #                 break
    #             f_out.write(line)
    
    # print(f"已保存 {num_lines:,} 行到: {output_file}")
    # print(f"新文件大小约为原文件的 {num_lines/total_lines*100:.2f}%")

def split_sft_dataset(input_file, output_file, num_lines=5000):
    """
    切分 SFT 数据集（BelleGroup）
    
    Args:
        input_file: 原始文件路径
        output_file: 输出文件路径
        num_lines: 保留的行数
    """
    print(f"正在处理: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"错误: 文件 {input_file} 不存在")
        return
    
    # 版本2：直接截取前N行，不进行所有行的遍历
    lines_written = 0
    with open(input_file, 'r', encoding='utf-8') as f_in:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for i, line in enumerate(f_in):
                # 增加一行判断
                if i >= num_lines:
                    break
                f_out.write(line)
                lines_written += 1

    print(f"已保存 {lines_written:,} 行到: {output_file}")

    # # 版本1：获取文件总行数
    # with open(input_file, 'r', encoding='utf-8') as f:
    #     total_lines = sum(1 for _ in f)
    # print(f"原始文件总行数: {total_lines:,}")
    
    # # 确定实际要保留的行数
    # num_lines = min(num_lines, total_lines)
    
    # # 读取并写入前 N 行
    # with open(input_file, 'r', encoding='utf-8') as f_in:
    #     with open(output_file, 'w', encoding='utf-8') as f_out:
    #         for i in range(num_lines):
    #             line = f_in.readline()
    #             if not line:
    #                 break
    #             f_out.write(line)
    
    # print(f"已保存 {num_lines:,} 行到: {output_file}")

if __name__ == "__main__":
    # ============ 配置参数 ============
    # 预训练数据集切分
    pretrain_input = "D:/workspace/deeplearning/d2cv/happy-llm/docs/dataset/mobvoi_seq_monkey_general_open_corpus.jsonl"
    pretrain_output = "D:/workspace/deeplearning/d2cv/happy-llm/docs/dataset/pretrain_mini_500.jsonl"
    
    # SFT 数据集切分
    # sft_input = "./datasets/BelleGroup/train_3.5M_CN.json"
    # sft_output = "./datasets/sft_mini.jsonl"
    
    # 保留的行数（可根据需要调整）
    NUM_LINES = 500  # 5000 条数据足够跑通流程
    
    print("=" * 50)
    print("开始切分数据集")
    print("=" * 50)
    
    # 切分预训练数据集
    print("\n1. 切分预训练数据集")
    split_pretrain_dataset(pretrain_input, pretrain_output, NUM_LINES)
    
    # # 切分 SFT 数据集
    # print("\n2. 切分 SFT 数据集")
    # split_sft_dataset(sft_input, sft_output, NUM_LINES)
    
    # print("\n" + "=" * 50)
    # print("全部完成！")
    # print(f"预训练小样本: {pretrain_output}")
    # print(f"SFT 小样本: {sft_output}")
    # print("=" * 50)