# 调试Prompt
import json
from config import CHUNKS_DATA_PATH

# --- LLM 指令 (Prompt) ---
# 我们从最终修复版里复制完全一样的Prompt模板
# 新的、正确的 EXTRACTION_PROMPT_TEMPLATE
EXTRACTION_PROMPT_TEMPLATE = """
你是一个顶级的、精通学术领域知识的AI专家。你的任务是从以下提供的文本片段中，抽取出结构化的知识三元组（Subject, Predicate, Object），中文称为（主语, 谓语, 宾语）。

请遵循以下规则：
1. 识别出的实体应该是具体的、有意义的名词或概念，例如模型名称、技术术语、机构、数据集等。
2. 谓词（Predicate）应该使用标准化的动词或关系短语，例如 'is_a' (是一种), 'developed_by' (由...开发), 'uses_method' (使用方法), 'is_part_of' (是...的一部分), 'has_property' (具有属性), 'contributes_to' (贡献于)。
3. 只从文本中包含的信息进行抽取，不要进行推理或使用外部知识。
4. 如果文本片段中没有可以抽取的有效三元组，请返回一个空的 "triples" 列表。
5. 你的回答必须是、且仅是一个格式正确的JSON对象，格式为：{{"triples": [{{"subject": "...", "predicate": "...", "object": "..."}}, ...]}}。不要在JSON前后添加任何解释性文字或代码块标记。

这是需要你处理的文本片段：
---
{text_chunk}
---
"""


def spy():
    print("--- 开始侦察第一个数据块 ---")

    # 1. 加载分块数据
    with open(CHUNKS_DATA_PATH, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)

    if not chunks_data:
        print("错误：chunks.json 为空或不存在。")
        return

    # 2. 获取第一个块
    first_chunk = chunks_data[0]
    chunk_text = first_chunk['text']
    chunk_id = first_chunk['chunk_id']

    print(f"\n[侦察报告] 第一个块的ID是: {chunk_id}")

    # 3. 打印原始的文本块内容
    print("\n--- [原始文本块内容] --- (请将下方```之间的所有内容复制)")
    print("```")
    print(chunk_text)
    print("```")

    # 4. 打印将要发送给API的完整Prompt
    full_prompt = EXTRACTION_PROMPT_TEMPLATE.format(text_chunk=chunk_text)
    print("\n--- [完整的API请求内容] ---")
    print(full_prompt)

    print("\n--- 侦察结束 ---")


if __name__ == "__main__":
    spy()