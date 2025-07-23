# src/04b_knowledge_extraction.py (ZhipuAI Version - Robust, with API Key)

import os
import json
from zhipuai import ZhipuAI
from tqdm import tqdm
import time

from config import PROCESSED_DATA_DIR, CHUNKS_DATA_PATH

# --- ZhipuAI API 配置 ---
# !!! 警告：API Key已硬编码。测试完成后请立即在官网作废此Key并使用.env方式管理新Key。
API_KEY = "a5528da8417645d8a4dfb71a9d30f140.7i6b5zfJDRYLgE9C"

CLIENT = ZhipuAI(api_key=API_KEY)
MODEL_NAME = "glm-4"

# --- 输出文件路径 ---
TRIPLES_DATA_PATH = PROCESSED_DATA_DIR / "triples.json"

# --- LLM 指令 (Prompt) ---
EXTRACTION_PROMPT_TEMPLATE = """
你是一个顶级的、精通学术领域知识的AI专家。你的任务是从以下提供的文本片段中，抽取出结构化的知识三元组（Subject, Predicate, Object），中文称为（主语, 谓语, 宾语）。

请遵循以下规则：
1. 识别出的实体应该是具体的、有意义的名词或概念，例如模型名称、技术术语、机构、数据集等。
2. 谓词（Predicate）应该使用标准化的动词或关系短语，例如 'is_a' (是一种), 'developed_by' (由...开发), 'uses_method' (使用方法), 'is_part_of' (是...的一部分), 'has_property' (具有属性), 'contributes_to' (贡献于)。
3. 只从文本中包含的信息进行抽取，不要进行推理或使用外部知识。
4. 如果文本片段中没有可以抽取的有效三元组，请返回一个空的 "triples" 列表。
5. 必须以JSON格式返回结果，格式为：{"triples": [{"subject": "...", "predicate": "...", "object": "..."}, ...]}

这是需要你处理的文本片段：
---
{text_chunk}
---
"""


def extract_triples_from_chunk(text_chunk, chunk_id):
    """调用ZhipuAI API从单个文本块中抽取三元组 (增强健壮性版本)"""
    # 增加一个重试机制
    for attempt in range(3):  # 最多重试3次
        try:
            response = CLIENT.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": EXTRACTION_PROMPT_TEMPLATE.format(text_chunk=text_chunk)
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            result_str = response.choices[0].message.content

            if not result_str:
                print(f"警告 (块ID: {chunk_id}): API返回为空内容。")
                return []

            # 尝试解析JSON
            result_json = json.loads(result_str)

            # 验证解析后的内容是否是我们期望的格式
            if isinstance(result_json, dict) and "triples" in result_json and isinstance(result_json["triples"], list):
                # 成功解析，并且格式正确，直接返回
                return result_json["triples"]
            else:
                print(f"警告 (块ID: {chunk_id}): API返回的JSON格式不符合预期。内容: {result_str}")
                return []

        except json.JSONDecodeError:
            print(
                f"警告-JSON解析失败 (块ID: {chunk_id}, 尝试次数 {attempt + 1}): API返回的不是有效的JSON。内容: {result_str}")
            time.sleep(2)  # 等待2秒再重试
            continue  # 继续下一次循环尝试

        except Exception as e:
            print(f"调用API时出错 (块ID: {chunk_id}, 尝试次数 {attempt + 1}): {e}")
            time.sleep(5)  # 发生其他错误时，等待更长时间
            continue

    print(f"错误 (块ID: {chunk_id}): 重试3次后仍然无法处理块。")
    return []


def process_all_chunks_for_triples(start_index=0, max_chunks=None):
    """遍历所有块，抽取三元组，并保存。增加参数用于测试。"""
    with open(CHUNKS_DATA_PATH, 'r', encoding='utf-8') as f:
        all_chunks = json.load(f)

    if max_chunks:
        chunks_to_process = all_chunks[start_index: start_index + max_chunks]
        print(f"--- 测试模式：只处理 {len(chunks_to_process)} 个块 (从索引 {start_index} 开始) ---")
    else:
        chunks_to_process = all_chunks
        print(f"开始从全部 {len(chunks_to_process)} 个文本块中抽取知识三元组...")

    # 如果要续传，可以先加载已有的triples文件
    all_triples = []
    if os.path.exists(TRIPLES_DATA_PATH):
        with open(TRIPLES_DATA_PATH, 'r', encoding='utf-8') as f:
            try:
                all_triples = json.load(f)
                print(f"已加载 {len(all_triples)} 个已有的三元组。")
            except json.JSONDecodeError:
                print("警告：现有的triples.json文件为空或已损坏，将创建新的。")

    # 创建一个已处理块的集合，用于续传
    processed_chunk_ids = {triple.get("source_chunk_id") for triple in all_triples}

    for chunk in tqdm(chunks_to_process, desc="抽取三元组"):
        chunk_id = chunk["chunk_id"]

        # 跳过已处理的块
        if chunk_id in processed_chunk_ids:
            continue

        chunk_text = chunk["text"]

        extracted_triples = extract_triples_from_chunk(chunk_text, chunk_id)

        if extracted_triples:
            for triple in extracted_triples:
                triple["source_chunk_id"] = chunk_id
                all_triples.append(triple)

        time.sleep(1)  # API调用频率限制

    with open(TRIPLES_DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_triples, f, indent=4, ensure_ascii=False)

    print(f"\n知识三元组抽取完成！总共获得 {len(all_triples)} 个三元组。")
    print(f"结果已保存至: {TRIPLES_DATA_PATH}")


if __name__ == "__main__":
    # 我们先测试20个块，看看效果
    process_all_chunks_for_triples(max_chunks=20)

    # --- 正式运行时，请使用下面这行 ---
    # process_all_chunks_for_triples()