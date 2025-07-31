# src/03_generate_paper_summaries_improved.py
import os
import json
import time
from zhipuai import ZhipuAI
from tqdm import tqdm
from pathlib import Path

from config import PROCESSED_DATA_DIR, TEXT_DATA_DIR

# API配置
API_KEY = "a5528da8417645d8a4dfb71a9d30f140.7i6b5zfJDRYLgE9C"
CLIENT = ZhipuAI(api_key=API_KEY)
MODEL_NAME = "glm-4"

# 概要保存路径
SUMMARIES_DIR = PROCESSED_DATA_DIR / "summaries"
SUMMARIES_DIR.mkdir(exist_ok=True)
SUMMARIES_DATA_PATH = PROCESSED_DATA_DIR / "document_summaries.json"

# 统一的文档概要生成提示词
DOCUMENT_SUMMARY_PROMPT = """
你是一位专业的文档分析专家。请分析以下文档，首先判断这是学术论文还是专利文档，然后严格基于原文内容提取结构化信息。

重要约束：
1. 只能提取原文中明确存在的信息，不得添加、推测或补充任何原文没有的内容
2. 如果某项信息在原文中不存在或不清楚，请在对应字段填写"未在原文中明确提及"
3. 所有提取的信息必须能在原文中找到对应的文字依据
4. 首先判断文档类型，然后按对应格式提取信息

文档原文：
---
{document_text}
---

如果是学术论文，请按以下JSON格式输出：
{{
  "document_type": "academic_paper",
  "title": "原文中的确切标题",
  "authors": ["原文中明确列出的作者姓名"],
  "main_topic": "原文中描述的主要研究领域",
  "research_problem": "原文中明确描述的要解决的问题",
  "methodology": "原文中提到的具体方法或算法名称",
  "key_innovations": ["原文中明确提到的创新点"],
  "experimental_results": "原文中的具体实验数据和结果",
  "conclusions": "原文结论部分的确切内容",
  "keywords": ["原文中的关键词或技术术语"],
  "application_domains": ["原文中明确提到的应用领域"],
  "technical_concepts": ["原文中出现的技术概念和算法名"],
  "performance_metrics": "原文中的具体性能数字和指标",
  "summary": "基于原文内容的200字概括"
}}

如果是专利文档，请按以下JSON格式输出：
{{
  "document_type": "patent",
  "title": "专利的确切名称",
  "inventors": ["原文中明确列出的发明人"],
  "patent_number": "专利号（如果有）",
  "application_domain": "专利的应用领域",
  "technical_problem": "专利要解决的技术问题",
  "technical_solution": "专利的技术方案描述",
  "key_innovations": ["专利的主要技术创新点"],
  "implementation_method": "具体实现方法或步骤",
  "technical_effects": "专利声称的技术效果",
  "keywords": ["从原文提取的关键技术词汇"],
  "application_scenarios": ["原文中提到的具体应用场景"],
  "technical_concepts": ["涉及的技术概念和方法"],
  "claims_summary": "权利要求的核心内容",
  "summary": "基于原文内容的200字专利概括"
}}

请只输出JSON格式的结果，不要添加任何解释。
"""


class DocumentSummaryGenerator:
    """文档概要生成器"""

    def __init__(self):
        self.max_text_length = 12000
        self.retry_count = 3
        self.success_count = 0
        self.total_count = 0

    def truncate_text_smartly(self, text):
        """智能截取文本，保留重要部分"""
        if len(text) <= self.max_text_length:
            return text

        paragraphs = text.split('\n\n')
        important_sections = []
        remaining_sections = []

        # 重要关键词（涵盖论文和专利）
        important_keywords = [
            'abstract', 'introduction', 'method', 'experiment', 'result', 'conclusion',
            '摘要', '引言', '方法', '实验', '结果', '结论',
            '技术领域', '背景技术', '发明内容', '技术方案', '有益效果',
            'technical field', 'background', 'summary', 'detailed description',
            '权利要求', 'claims', '实施例', 'embodiment'
        ]

        for para in paragraphs:
            para_lower = para.lower()
            if any(keyword in para_lower for keyword in important_keywords):
                important_sections.append(para)
            else:
                remaining_sections.append(para)

        # 组装文本
        result_text = '\n\n'.join(important_sections)

        for para in remaining_sections:
            if len(result_text + '\n\n' + para) <= self.max_text_length:
                result_text += '\n\n' + para
            else:
                break

        return result_text

    def generate_summary_for_document(self, text, file_id):
        """为单个文档生成概要"""

        # 智能截取文本
        processed_text = self.truncate_text_smartly(text)

        for attempt in range(self.retry_count):
            try:
                response = CLIENT.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {
                            "role": "user",
                            "content": DOCUMENT_SUMMARY_PROMPT.format(document_text=processed_text)
                        }
                    ],
                    temperature=0.05,
                    max_tokens=1500,
                )

                result_str = response.choices[0].message.content.strip()
                cleaned_str = result_str.replace("```json", "").replace("```", "").strip()

                if not cleaned_str:
                    print(f"空返回: {file_id}, 尝试 {attempt + 1}")
                    continue

                # 解析JSON
                summary_json = json.loads(cleaned_str)

                # 验证必要字段
                required_fields = ['title', 'summary', 'document_type']
                if all(field in summary_json for field in required_fields):
                    doc_type = summary_json.get('document_type')
                    print(f"成功生成概要: {file_id} ({doc_type})")
                    return summary_json
                else:
                    print(f"概要格式不完整: {file_id}, 尝试 {attempt + 1}")
                    continue

            except json.JSONDecodeError as e:
                print(f"JSON解析失败: {file_id}, 尝试 {attempt + 1}, 错误: {e}")
                time.sleep(2)
                continue
            except Exception as e:
                print(f"API调用失败: {file_id}, 尝试 {attempt + 1}, 错误: {e}")
                time.sleep(5)
                continue

        print(f"最终失败: {file_id}")
        return None

    def save_individual_summary(self, file_id, summary_data):
        """保存单个文档的概要到独立文件"""
        doc_type = summary_data.get('document_type', 'unknown')
        summary_file = SUMMARIES_DIR / f"{file_id}_{doc_type}_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)


def main():
    """主函数：批量生成所有文档概要"""
    print("=" * 60)
    print("文档概要生成系统")
    print("自动识别论文和专利文档类型")
    print("严格基于原文内容提取信息")
    print("=" * 60)

    metadata_path = PROCESSED_DATA_DIR / "metadata.json"

    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print("错误：metadata.json文件未找到")
        return

    generator = DocumentSummaryGenerator()

    # 使用清洗后的文件
    print("使用清洗后的文件生成概要...")
    available_files = {k: v for k, v in metadata.items()
                       if v.get("cleaning_status") == "cleaned"
                       and "cleaned_text_path" in v}

    if not available_files:
        print("没有找到清洗后的文件，请先完成LLM清洗")
        return

    print(f"共找到 {len(available_files)} 个可处理的文档")

    # 加载已有的概要数据
    all_summaries = {}
    if os.path.exists(SUMMARIES_DATA_PATH):
        try:
            with open(SUMMARIES_DATA_PATH, 'r', encoding='utf-8') as f:
                all_summaries = json.load(f)
            print(f"加载了 {len(all_summaries)} 个已有概要")
        except:
            print("概要文件损坏，将创建新的")
            all_summaries = {}

    # 统计信息
    stats = {"patent": 0, "academic_paper": 0, "failed": 0}

    # 开始生成概要
    for file_id, info in tqdm(available_files.items(), desc="生成概要"):
        # 跳过已处理的文件
        if file_id in all_summaries:
            print(f"跳过已有概要: {file_id}")
            continue

        # 获取文本文件路径
        text_path = info.get("cleaned_text_path")

        if not text_path or not os.path.exists(text_path):
            print(f"文件不存在: {file_id}")
            continue

        try:
            # 读取文本
            with open(text_path, 'r', encoding='utf-8') as f:
                document_text = f.read()

            if len(document_text.strip()) < 500:
                print(f"文本过短，跳过: {file_id}")
                continue

            # 生成概要
            summary = generator.generate_summary_for_document(document_text, file_id)

            if summary:
                # 添加元信息
                summary["file_id"] = file_id
                summary["source_type"] = "cleaned"
                summary["text_length"] = len(document_text)
                summary["generation_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

                # 保存到总集合
                all_summaries[file_id] = summary

                # 保存单独文件
                generator.save_individual_summary(file_id, summary)

                generator.success_count += 1
                doc_type = summary.get('document_type', 'unknown')
                stats[doc_type] = stats.get(doc_type, 0) + 1

                # 显示概要预览
                print(f"{file_id} ({doc_type}) 概要预览:")
                print(f"   标题: {summary.get('title', '未知')}")

                if doc_type == "patent":
                    inventors = summary.get('inventors', [])
                    print(f"   发明人: {', '.join(inventors[:2]) if inventors else '未知'}")
                else:
                    authors = summary.get('authors', [])
                    print(f"   作者: {', '.join(authors[:2]) if authors else '未知'}")

                print()
            else:
                stats["failed"] += 1
                print(f"生成失败: {file_id}")

            generator.total_count += 1

            # 每处理5个文件保存一次
            if generator.total_count % 5 == 0:
                with open(SUMMARIES_DATA_PATH, 'w', encoding='utf-8') as f:
                    json.dump(all_summaries, f, indent=2, ensure_ascii=False)
                print(f"中间保存... 已处理 {generator.total_count}")

            # API调用间隔
            time.sleep(3)

        except Exception as e:
            stats["failed"] += 1
            print(f"处理异常: {file_id}, {e}")
            continue

    # 最终保存
    with open(SUMMARIES_DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    # 显示统计结果
    print(f"\n概要生成完成！")
    print(f"处理统计:")
    print(f"   论文文档: {stats.get('academic_paper', 0)} 个")
    print(f"   专利文档: {stats.get('patent', 0)} 个")
    print(f"   处理失败: {stats.get('failed', 0)} 个")
    if generator.total_count > 0:
        print(
            f"   总成功率: {generator.success_count}/{generator.total_count} ({generator.success_count / generator.total_count * 100:.1f}%)")
    print(f"概要保存在: {SUMMARIES_DATA_PATH}")
    print(f"单独文件保存在: {SUMMARIES_DIR}")


if __name__ == "__main__":
    main()