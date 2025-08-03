#用于处理返回的json无法解析的情况
import os
import json
import time
from zhipuai import ZhipuAI
from tqdm import tqdm
from pathlib import Path

from config import PROCESSED_DATA_DIR

# API配置
API_KEY = "838cc9e6876a4fea971b3728af105b56.1KDgfLzNHnfllnhb"
CLIENT = ZhipuAI(api_key=API_KEY)
MODEL_NAME = "glm-4"

# 路径配置
STRUCTURAL_DIR = PROCESSED_DATA_DIR / "structural_insights"
STRUCTURAL_DATA_PATH = STRUCTURAL_DIR / "structural_insights.json"
INDIVIDUAL_INSIGHTS_DIR = STRUCTURAL_DIR / "individual"

# 极简版提示词 - 避免复杂格式
SIMPLE_PROMPT = """请从以下文档中提取基本信息，以JSON格式输出。
如果没有找到信息的话 该字段可以填写 未找到该信息 
只返回json格式的信息就可以
文档内容：
{document_text}

请提取以下信息：
- 标题
- 作者
- 主要内容
- 关键词

输出JSON格式：
{{
  "title": "文档标题",
  "authors": ["作者1", "作者2"],
  "main_content": "主要内容描述",
  "keywords": ["关键词1", "关键词2"]
}}"""


def safe_json_parse(json_string, file_id):
    """极简JSON解析"""
    try:
        # 基本清理
        cleaned = json_string.strip()
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()

        # 直接解析
        return json.loads(cleaned)
    except:
        print(f"JSON解析失败: {file_id}")
        print(f"返回内容: {repr(json_string)}")
        return None


def extract_simple_insights(text, file_id, source_type):
    """极简版洞察提取"""

    # 截取文本到安全长度
    if len(text) > 3000:
        processed_text = text[:3000]
    else:
        processed_text = text

    prompt = SIMPLE_PROMPT.format(document_text=processed_text)

    print(f"处理: {file_id}")
    print(f"文本长度: {len(processed_text)} 字符")
    print(f"提示词长度: {len(prompt)} 字符")

    for attempt in range(3):
        try:
            print(f"API调用尝试 {attempt + 1}/3...")

            # 使用更保守的参数
            response = CLIENT.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=800,  # 减少token数
                # 移除停止词
            )

            result_str = response.choices[0].message.content.strip()

            print(f"API返回长度: {len(result_str)} 字符")
            print(f"API返回内容: {repr(result_str[:200])}...")

            if len(result_str) < 10:  # 返回内容太短
                print(f"返回内容过短，重试...")
                time.sleep(2)
                continue

            # 解析JSON
            parsed = safe_json_parse(result_str, file_id)

            if parsed:
                # 转换为标准格式
                standard_format = {
                    "document_metadata": {
                        "document_type": "unknown",
                        "title": parsed.get("title", "原文无此信息"),
                        "authors": parsed.get("authors", []),
                    },
                    "main_content": {
                        "description": parsed.get("main_content", "原文无此信息"),
                        "keywords": parsed.get("keywords", [])
                    },
                    "extraction_metadata": {
                        "file_id": file_id,
                        "source_type": source_type,
                        "extraction_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "extraction_method": "simple_supplement",
                        "text_length": len(text)
                    }
                }

                print(f"成功提取: {file_id}")
                return standard_format
            else:
                print(f"JSON解析失败，重试...")
                time.sleep(2)
                continue

        except Exception as e:
            print(f"API调用失败: {e}")
            time.sleep(3)
            continue

    print(f"最终失败: {file_id}")
    return None


def find_missing_documents():
    """找出缺失的文档"""
    metadata_path = PROCESSED_DATA_DIR / "metadata.json"
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    existing_insights = {}
    if os.path.exists(STRUCTURAL_DATA_PATH):
        try:
            with open(STRUCTURAL_DATA_PATH, 'r', encoding='utf-8') as f:
                existing_insights = json.load(f)
        except:
            existing_insights = {}

    missing_docs = {}
    for file_id, info in metadata.items():
        if file_id in existing_insights:
            continue

        # 优先使用清洗后文件
        if (info.get("cleaning_status") == "cleaned" and
                "cleaned_text_path" in info and
                os.path.exists(info["cleaned_text_path"])):
            try:
                with open(info["cleaned_text_path"], 'r', encoding='utf-8') as f:
                    text = f.read()
                if len(text.strip()) >= 100:  # 降低最小长度要求
                    missing_docs[file_id] = {
                        'text_path': info["cleaned_text_path"],
                        'source_type': "cleaned"
                    }
                    continue
            except:
                pass

        # 使用原始文件
        if (info.get("status") == "text_extracted" and
                "text_path" in info and
                os.path.exists(info["text_path"])):
            try:
                with open(info["text_path"], 'r', encoding='utf-8') as f:
                    text = f.read()
                if len(text.strip()) >= 100:
                    missing_docs[file_id] = {
                        'text_path': info["text_path"],
                        'source_type': "basic"
                    }
            except:
                pass

    return missing_docs, existing_insights


def main():
    """主函数 - 极简版补充"""
    print("=" * 60)
    print("极简版结构化洞察补充")
    print("=" * 60)

    missing_docs, existing_insights = find_missing_documents()

    if not missing_docs:
        print("所有文档都已处理完成！")
        return

    print(f"需要处理 {len(missing_docs)} 个文档")
    print("-" * 60)

    success_count = 0

    for i, (file_id, file_info) in enumerate(missing_docs.items()):
        print(f"\n[{i + 1}/{len(missing_docs)}] 处理: {file_id}")

        try:
            # 读取文本
            with open(file_info['text_path'], 'r', encoding='utf-8') as f:
                text = f.read()

            # 提取洞察
            insights = extract_simple_insights(text, file_id, file_info['source_type'])

            if insights:
                # 保存
                existing_insights[file_id] = insights

                # 保存单独文件
                insights_file = INDIVIDUAL_INSIGHTS_DIR / f"{file_id}_simple_supplement.json"
                with open(insights_file, 'w', encoding='utf-8') as f:
                    json.dump(insights, f, indent=2, ensure_ascii=False)

                success_count += 1
                print(f"成功处理!")
            else:
                print(f"处理失败!")

            # 每次都保存进度
            with open(STRUCTURAL_DATA_PATH, 'w', encoding='utf-8') as f:
                json.dump(existing_insights, f, indent=2, ensure_ascii=False)

            # 短暂等待
            time.sleep(2)

        except Exception as e:
            print(f"异常: {e}")
            continue

    print(f"\n" + "=" * 60)
    print("处理完成!")
    print(f"成功: {success_count}/{len(missing_docs)}")
    print(f"总洞察数: {len(existing_insights)}")
    print("=" * 60)


if __name__ == "__main__":
    main()