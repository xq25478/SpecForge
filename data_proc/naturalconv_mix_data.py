# 查看数据
# import json
# import codecs
# dialog_list = json.loads(codecs.open("dialog_release.json", "r", "utf-8").read())

# i=0

# for dialog in dialog_list:
#     i += 1
#     # print(dialog)
# print(i)


# 划分数据
import json

def convert_json_file(input_json_path, train_txt_path, train_jsonl_path, val_jsonl_path):
    len_train = len_test = 0
    # 读取 train.txt 中的 dialog_id 列表
    with open(train_txt_path, 'r', encoding='utf-8') as f:
        train_dialog_ids = set(line.strip() for line in f if line.strip())

    # 读取原始 JSON 文件（假设是包含多个对象的列表）
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    # 准备写入 jsonl 文件
    with open(train_jsonl_path, 'w', encoding='utf-8') as train_f, \
         open(val_jsonl_path, 'w', encoding='utf-8') as val_f:

        for item in data_list:
            dialog_id = item['dialog_id']
            content = item['content']

            # 构建 conversations 列表
            conversations = []
            for i, text in enumerate(content):
                speaker = "user" if i % 2 == 0 else "assistant"
                conversations.append({"role": speaker, "content": text})

            # 构建新格式
            new_item = {
                "id": dialog_id,
                "conversations": conversations
            }

            # 判断写入哪个文件
            if dialog_id in train_dialog_ids:
                target_file = train_f #if dialog_id in train_dialog_ids else val_f
                len_train += 1
            else:
                target_file = val_f
                len_test += 1
            target_file.write(json.dumps(new_item, ensure_ascii=False) + '\n')

    print(f"转换完成！")
    print(f"训练集保存至: {train_jsonl_path}")
    print(f"验证集保存至: {val_jsonl_path}")
    print(f"训练数据集有{len_train}条")
    print(f"测试数据集有{len_test}条")

import json
import random

def random_sample_jsonl(input_file, output_file, sample_size):
    # 读取所有数据
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 检查是否足够
    if len(lines) < sample_size:
        raise ValueError(f"文件只有 {len(lines)} 行，不足 {sample_size} 行可采样")

    # 随机采样
    sampled_lines = random.sample(lines, sample_size)

    # 写出到新文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in sampled_lines:
            f.write(line)

    print(f"已从 {input_file} 随机采样 {sample_size} 条数据，保存至 {output_file}")


# 使用示例
if __name__ == "__main__":
    # 划分中文数据集
    # convert_json_file(
    #     input_json_path='/mnt/modelops/dataset/dialog_release.json',       # 输入的原始 JSON 文件
    #     train_txt_path='/mnt/modelops/dataset/train.txt',         # 包含 dialog_id 的训练集 ID 列表
    #     train_jsonl_path='/mnt/modelops/dataset/train.jsonl',     # 输出训练集（JSONL 格式）
    #     val_jsonl_path='/mnt/modelops/dataset/test.jsonl'          # 输出验证集（JSONL 格式）
    # )

    # 划分ultrachat19375条
    random_sample_jsonl('/mnt/modelops/train/eagle3/baseline_ultrachat_sft_train_only/data/ultrachat_sft_train.jsonl', '/mnt/modelops/487922/dataset/ultrachat_sampled_77500.jsonl', 77500)
