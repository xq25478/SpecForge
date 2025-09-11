import json
import sys

def convert_json_to_jsonl(input_json_path, output_jsonl_path):
    """
    将原始JSON文件（包含多个样本的列表）转换为JSONL格式
    每行输出一个样本：{"id": "...", "conversations": [...]}
    """
    try:
        # 1. 读取输入的JSON文件
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)  # 应该是一个列表，每个元素是一个样本

        # 2. 确保是列表格式
        if not isinstance(data, list):
            raise ValueError("JSON文件的根结构必须是一个数组（list），每个元素是一个样本。")

        # 3. 处理每个样本并写入JSONL
        with open(output_jsonl_path, 'w', encoding='utf-8') as f_out:
            for entry in data:
                # 提取id，如果没有则生成一个默认id
                sample_id = entry.get("id", "unknown_id")

                messages = entry.get("messages", [])
                conversations = []

                for i, msg in enumerate(messages):
                    if "message" not in msg:
                        continue  # 跳过无效消息

                    # 判断角色：奇数轮为 user，偶数轮为 assistant（从0开始）
                    role = "user" if i % 2 == 0 else "assistant"
                    content = msg["message"].strip()

                    # 如果是 assistant 的回复，并且有 attrs，尝试补充知识
                    if role == "assistant" and "attrs" in msg and len(msg["attrs"]) > 0:
                        attr = msg["attrs"][0]
                        attrvalue = attr.get("attrvalue", "").strip()
                        # 如果 attrvalue 不为空，且未包含在原回复中，可以追加
                        if attrvalue and attrvalue not in content:
                            content = content.rstrip("。!?") + "。" + attrvalue

                    conversations.append({
                        "role": role,
                        "content": content
                    })

                # 构建输出样本
                output_sample = {
                    "id": sample_id,
                    "conversations": conversations
                }

                # 写入一行 JSONL
                f_out.write(json.dumps(output_sample, ensure_ascii=False) + "\n")

        print(f"✅ 转换完成！已将 {len(data)} 个样本写入：{output_jsonl_path}")

    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 {input_json_path}")
    except json.JSONDecodeError as e:
        print(f"❌ 错误：JSON解析失败：{e}")
    except Exception as e:
        print(f"❌ 发生未知错误：{e}")


# ========================
# 主程序入口
# ========================

if __name__ == "__main__":
    # 可以通过命令行传参，也可以直接修改路径
    input_file = "/mnt/modelops/dataset/kvconv/data/travel/train.json"        # 输入文件路径
    output_file = "/mnt/modelops/dataset/kvconv/data/traveloutput.jsonl"    # 输出文件路径

    convert_json_to_jsonl(input_file, output_file)
