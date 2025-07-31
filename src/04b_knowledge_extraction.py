# src/04b_knowledge_extraction.py (最终优化版)

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

# --- LLM 指令 (Prompt) - 优化版 ---
EXTRACTION_PROMPT_TEMPLATE = """
你是一个顶级的、精通学术领域知识的AI专家。你的任务是从以下提供的文本片段中，抽取出结构化的知识三元组（Subject, Predicate, Object），中文称为（主语, 谓语, 宾语）。

请遵循以下规则：
1. 识别出的实体应该是具体的、有意义的名词或概念。优先关注那些描述技术、模型、理论、实验结果的核心知识。作者、机构等元数据信息如果明确出现，也应抽取。
2. 对抽取出的实体（主语和宾语）进行标准化和清洗。移除任何由于PDF解析产生的格式噪音，例如多余的换行符、连字符、特殊符号（如'\\', '^'）等。例如，如果文本中是 "Ruifan L\\ Cong Wang^"，你应该在三元组中将其修正为 "Ruifan Li, Cong Wang"。
3. 谓词（Predicate）应该使用标准化的动词或关系短语，例如 'is_a' (是一种), 'developed_by' (由...开发), 'uses_method' (使用方法), 'is_part_of' (是...的一部分), 'has_property' (具有属性), 'contributes_to' (贡献于)。
4. 只从文本中包含的信息进行抽取，但可以在清洗实体时进行合理的修正，不要进行过度推理。
5. 如果文本片段中没有可以抽取的有效三元组，请返回一个空的 "triples" 列表。
6. 你的回答必须是、且仅是一个格式正确的JSON对象，格式为：{{"triples": [{{"subject": "...", "predicate": "...", "object": "..."}}, ...]}}。不要在JSON前后添加任何解释性文字或代码块标记。

这是需要你处理的文本片段：
---
{text_chunk}
---
"""


def extract_triples_from_chunk(text_chunk, chunk_id):
    """调用ZhipuAI API从单个文本块中抽取三元组 (健壮优化版)"""
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
                temperature=0.0,  # <-- 关键优化：设为0.0以获得最确定的结果
            )
            result_str = response.choices[0].message.content

            # 清理模型可能返回的Markdown代码块标记
            cleaned_str = result_str.strip().replace("```json", "").replace("```", "").strip()

            if not cleaned_str:
                return []

            result_json = json.loads(cleaned_str)

            if isinstance(result_json, dict) and "triples" in result_json:
                return result_json.get("triples", [])
            else:
                print(f"警告 (块ID: {chunk_id}): 返回的JSON格式不符预期。内容: {cleaned_str}")
                return []

        except json.JSONDecodeError:
            print(f"警告-JSON解析失败 (块ID: {chunk_id}, 尝试 {attempt + 1}): API返回的不是有效JSON。内容: {result_str}")
            time.sleep(2)
            continue

        except Exception as e:
            print(f"调用API时出错 (块ID: {chunk_id}, 尝试 {attempt + 1}): {repr(e)}")
            time.sleep(5)
            continue

    print(f"错误 (块ID: {chunk_id}): 重试3次后仍然无法处理。")
    return []


def process_all_chunks_for_triples(start_index=0, max_chunks=None):
    """遍历所有块，抽取三元组，并保存。"""
    with open(CHUNKS_DATA_PATH, 'r', encoding='utf-8') as f:
        all_chunks = json.load(f)

    if max_chunks:
        chunks_to_process = all_chunks[start_index: start_index + max_chunks]
        print(f"--- 测试模式：只处理 {len(chunks_to_process)} 个块 (从索引 {start_index} 开始) ---")
    else:
        chunks_to_process = all_chunks
        print(f"开始从全部 {len(chunks_to_process)} 个文本块中抽取知识三元组...")

    all_triples = []
    if os.path.exists(TRIPLES_DATA_PATH) and start_index == 0:
        try:
            with open(TRIPLES_DATA_PATH, 'r', encoding='utf-8') as f:
                all_triples = json.load(f)
            print(f"已加载 {len(all_triples)} 个已有的三元组，将进行续传。")
        except (json.JSONDecodeError, FileNotFoundError):
            print("警告：现有的triples.json文件为空或已损坏，将创建新的。")
            all_triples = []

    processed_chunk_ids = {triple.get("source_chunk_id") for triple in all_triples}

    for chunk in tqdm(chunks_to_process, desc="抽取三元组"):
        chunk_id = chunk["chunk_id"]
        if chunk_id in processed_chunk_ids:
            continue

        chunk_text = chunk["text"]
        extracted_triples = extract_triples_from_chunk(chunk_text, chunk_id)
        if extracted_triples:
            for triple in extracted_triples:
                triple["source_chunk_id"] = chunk_id
            all_triples.extend(extracted_triples)

        # 每次处理后都保存，确保中断后能恢复
        with open(TRIPLES_DATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(all_triples, f, indent=4, ensure_ascii=False)

        time.sleep(1)  # 遵循API调用频率限制

    print(f"\n知识三元组抽取完成！总共获得 {len(all_triples)} 个三元组。")
    print(f"结果已保存至: {TRIPLES_DATA_PATH}")


if __name__ == "__main__":

    process_all_chunks_for_triples()