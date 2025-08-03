# src/supplement_structural_insights.py
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

# 文档类型检测关键词
EXPERIMENTAL_KEYWORDS = [
    'experiment', 'evaluation', 'dataset', 'performance', 'accuracy', 'precision',
    'recall', 'f1-score', 'training', 'testing', 'validation', 'benchmark',
    '实验', '评估', '数据集', '性能', '准确率', '精确率', '召回率', '训练', '测试', '验证'
]

THEORETICAL_KEYWORDS = [
    'theorem', 'proof', 'lemma', 'proposition', 'mathematical', 'theoretical',
    'analysis', 'formal', 'definition', 'axiom',
    '定理', '证明', '引理', '命题', '数学', '理论', '分析', '形式化', '定义', '公理'
]

PATENT_KEYWORDS = [
    'patent', 'invention', 'claim', 'embodiment', 'technical solution',
    '专利', '发明', '权利要求', '实施例', '技术方案', '发明人', '申请人'
]

# 简化但稳定的提示词模板
SUPPLEMENT_EXTRACTION_PROMPT = """
你是一位严格的学术文献分析专家。请从以下文档中提取结构化信息。

重要约束：
1. 只能提取原文明确存在的信息，不能编造
2. 如果信息不存在，填写"原文无此信息"
3. 只输出JSON格式，不要任何额外文字

文档原文：
---
{document_text}
---

请根据文档类型({doc_type})按以下JSON格式输出：

{format_template}

只输出JSON，不要解释：
"""

# 实验性论文格式
EXPERIMENTAL_FORMAT = """{
  "document_metadata": {
    "document_type": "experimental_paper",
    "title": "原文确切标题或'原文无此信息'",
    "authors": ["原文明确列出的作者，如无则为空数组"],
    "institutions": ["原文明确提到的机构，如无则为空数组"]
  },
  "technical_relationships": {
    "base_methods": [{"method_name": "原文明确的基础方法或'原文无此信息'", "relationship_type": "基于|改进|扩展或'原文无此信息'"}],
    "compared_methods": [{"method_name": "原文明确对比的方法或'原文无此信息'", "comparison_result": "原文明确的对比结果或'原文无此信息'"}]
  },
  "experimental_setup": {
    "datasets_used": [{"dataset_name": "原文明确的数据集名或'原文无此信息'", "dataset_description": "原文描述或'原文无此信息'"}],
    "evaluation_metrics": ["原文明确的评估指标，如无则为空数组"],
    "baseline_methods": ["原文明确的基准方法，如无则为空数组"]
  },
  "performance_results": {
    "quantitative_results": [{"metric_name": "原文明确的指标或'原文无此信息'", "our_result": "我们的结果或'原文无此信息'", "baseline_result": "基准结果或'原文无此信息'"}]
  },
  "innovation_analysis": {
    "stated_contributions": ["原文明确声明的贡献，如无则为空数组"],
    "stated_novelty": ["原文明确声明的新颖性，如无则为空数组"]
  },
  "limitations": {
    "acknowledged_limitations": ["原文明确承认的局限性，如无则为空数组"]
  }
}"""

# 理论性论文格式
THEORETICAL_FORMAT = """{
  "document_metadata": {
    "document_type": "theoretical_paper",
    "title": "原文确切标题或'原文无此信息'",
    "authors": ["原文明确列出的作者，如无则为空数组"],
    "institutions": ["原文明确提到的机构，如无则为空数组"]
  },
  "theoretical_contributions": {
    "main_theoretical_results": ["原文明确的理论结果，如无则为空数组"],
    "theorems_proposed": ["原文提出的定理，如无则为空数组"],
    "mathematical_models": ["原文的数学模型，如无则为空数组"]
  },
  "technical_relationships": {
    "builds_upon": ["原文明确基于的理论，如无则为空数组"],
    "extends": ["原文明确扩展的理论，如无则为空数组"],
    "relates_to": ["原文明确相关的理论，如无则为空数组"]
  },
  "innovation_analysis": {
    "theoretical_novelty": ["原文声明的理论新颖性，如无则为空数组"],
    "mathematical_contributions": ["原文声明的数学贡献，如无则为空数组"]
  },
  "applications": {
    "potential_applications": ["原文提到的潜在应用，如无则为空数组"],
    "application_domains": ["原文提到的应用领域，如无则为空数组"]
  }
}"""

# 专利格式
PATENT_FORMAT = """{
  "document_metadata": {
    "document_type": "patent",
    "title": "专利确切名称或'原文无此信息'",
    "inventors": ["原文明确列出的发明人，如无则为空数组"],
    "applicant": "申请人信息或'原文无此信息'",
    "patent_number": "专利号或'原文无此信息'"
  },
  "technical_solution": {
    "technical_problem": "原文描述的技术问题或'原文无此信息'",
    "solution_overview": "原文的技术方案概述或'原文无此信息'",
    "key_technical_features": ["原文明确的关键技术特征，如无则为空数组"]
  },
  "implementation": {
    "embodiments": ["原文描述的实施例，如无则为空数组"],
    "technical_effects": ["原文声称的技术效果，如无则为空数组"]
  },
  "application_scope": {
    "application_fields": ["原文明确的应用领域，如无则为空数组"],
    "use_scenarios": ["原文描述的使用场景，如无则为空数组"]
  }
}"""

# 未知类型格式
UNKNOWN_FORMAT = """{
  "document_metadata": {
    "document_type": "unknown",
    "title": "原文确切标题或'原文无此信息'",
    "authors_or_creators": ["原文明确的作者，如无则为空数组"]
  },
  "main_content": {
    "stated_purpose": "原文明确的目的或'原文无此信息'",
    "key_concepts": ["原文明确的关键概念，如无则为空数组"],
    "main_methods": ["原文明确的方法，如无则为空数组"]
  },
  "relationships": {
    "references_to": ["原文明确引用的其他工作，如无则为空数组"],
    "builds_on": ["原文明确基于的工作，如无则为空数组"]
  }
}"""


def safe_json_parse(json_string, file_id):
    """安全的JSON解析（增强调试版本）"""
    cleaned = json_string.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"JSON解析失败: {file_id}, 错误: {e}")

        # 显示原始内容用于调试
        print(f"原始返回内容长度: {len(json_string)} 字符")
        print(f"清理后内容长度: {len(cleaned)} 字符")
        print("原始返回内容预览:")
        print("=" * 60)
        print(repr(json_string[:500]))  # 使用repr显示特殊字符
        print("=" * 60)
        print("清理后内容预览:")
        print("-" * 60)
        print(repr(cleaned[:500]))  # 使用repr显示特殊字符
        print("-" * 60)

        # 尝试提取第一个完整的JSON对象
        try:
            start_pos = cleaned.find('{')
            if start_pos == -1:
                print("未找到JSON开始标记 '{'")
                return None

            brace_count = 0
            end_pos = start_pos

            for i, char in enumerate(cleaned[start_pos:], start_pos):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i + 1
                        break

            first_json = cleaned[start_pos:end_pos]
            print(f"提取的JSON片段:")
            print("-" * 30)
            print(repr(first_json[:200]))
            print("-" * 30)
            return json.loads(first_json)

        except json.JSONDecodeError as e2:
            print(f"提取第一个JSON也失败: {file_id}, 错误: {e2}")
            return None


class SupplementExtractor:
    """补充提取器（增强调试版本）"""

    def __init__(self):
        self.max_text_length = 12000
        self.retry_count = 3
        self.success_count = 0
        self.total_count = 0

    def detect_document_type(self, text):
        """检测文档类型"""
        text_lower = text.lower()

        experimental_score = sum(1 for kw in EXPERIMENTAL_KEYWORDS if kw in text_lower)
        theoretical_score = sum(1 for kw in THEORETICAL_KEYWORDS if kw in text_lower)
        patent_score = sum(1 for kw in PATENT_KEYWORDS if kw in text_lower)

        if patent_score >= 3:
            return "patent"
        elif experimental_score >= 5:
            return "experimental_paper"
        elif theoretical_score >= 3 and experimental_score < 3:
            return "theoretical_paper"
        else:
            return "unknown"

    def get_format_template(self, doc_type):
        """获取格式模板"""
        if doc_type == "experimental_paper":
            return EXPERIMENTAL_FORMAT
        elif doc_type == "theoretical_paper":
            return THEORETICAL_FORMAT
        elif doc_type == "patent":
            return PATENT_FORMAT
        else:
            return UNKNOWN_FORMAT

    def truncate_text(self, text):
        """截取文本"""
        if len(text) <= self.max_text_length:
            return text
        return text[:self.max_text_length]

    def extract_insights(self, text, file_id, source_type):
        """提取结构化洞察（增强调试版本）"""

        doc_type = self.detect_document_type(text)
        print(f"处理: {file_id} -> {doc_type} ({source_type})")
        print(f"原始文本长度: {len(text)} 字符")

        processed_text = self.truncate_text(text)
        print(f"处理后文本长度: {len(processed_text)} 字符")

        format_template = self.get_format_template(doc_type)

        prompt = SUPPLEMENT_EXTRACTION_PROMPT.format(
            document_text=processed_text,
            doc_type=doc_type,
            format_template=format_template
        )

        print(f"提示词长度: {len(prompt)} 字符")

        for attempt in range(self.retry_count):
            try:
                print(f"API调用尝试 {attempt + 1}/{self.retry_count}...")

                response = CLIENT.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.01,
                    max_tokens=1500,
                    stop=["\n\n注意", "\n\n说明", "```"]
                )

                result_str = response.choices[0].message.content.strip()

                # 详细显示API返回信息
                print(f"API调用成功!")
                print(f"返回内容长度: {len(result_str)} 字符")
                print(f"返回内容类型: {type(result_str)}")
                print("API原始返回内容:")
                print("=" * 80)
                print(repr(result_str))  # 使用repr显示所有特殊字符
                print("=" * 80)
                print("API返回内容（直接显示）:")
                print("-" * 80)
                print(result_str)
                print("-" * 80)

                # 检查是否为空
                if not result_str or result_str.isspace():
                    print(f"⚠️ API返回内容为空或只包含空白字符")
                    print(f"尝试 {attempt + 1} 失败，继续重试...")
                    time.sleep(3)
                    continue

                insights_json = safe_json_parse(result_str, file_id)

                if insights_json is None:
                    print(f"JSON解析失败: {file_id}, 尝试 {attempt + 1}")
                    time.sleep(3)
                    continue

                # 添加元信息
                insights_json['extraction_metadata'] = {
                    'file_id': file_id,
                    'detected_doc_type': doc_type,
                    'source_type': source_type,
                    'extraction_time': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'text_length': len(text),
                    'processed_text_length': len(processed_text),
                    'extraction_method': 'supplement_script'
                }

                print(f"成功提取: {file_id} ({doc_type})")
                return insights_json

            except Exception as e:
                print(f"API调用异常: {file_id}, 尝试 {attempt + 1}, 错误类型: {type(e).__name__}")
                print(f"错误详情: {str(e)}")

                # 显示更多调试信息
                import traceback
                print("完整错误堆栈:")
                print(traceback.format_exc())

                time.sleep(5)
                continue

        print(f"最终失败: {file_id}")
        return None

    def save_individual_insight(self, file_id, insights_data):
        """保存单个洞察"""
        doc_type = insights_data.get('extraction_metadata', {}).get('detected_doc_type', 'unknown')
        source_type = insights_data.get('extraction_metadata', {}).get('source_type', 'unknown')
        insights_file = INDIVIDUAL_INSIGHTS_DIR / f"{file_id}_{doc_type}_{source_type}_supplement.json"
        with open(insights_file, 'w', encoding='utf-8') as f:
            json.dump(insights_data, f, indent=2, ensure_ascii=False)


def find_missing_documents():
    """找出缺失的文档"""
    print("=" * 60)
    print("补充结构化洞察提取（调试版本）")
    print("=" * 60)

    # 加载metadata
    metadata_path = PROCESSED_DATA_DIR / "metadata.json"
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print("错误：metadata.json文件未找到")
        return [], {}

    # 加载已有洞察
    existing_insights = {}
    if os.path.exists(STRUCTURAL_DATA_PATH):
        try:
            with open(STRUCTURAL_DATA_PATH, 'r', encoding='utf-8') as f:
                existing_insights = json.load(f)
        except:
            existing_insights = {}

    print(f"总文档数: {len(metadata)}")
    print(f"已有洞察: {len(existing_insights)}")

    # 找出可处理但缺失的文档
    missing_docs = {}

    for file_id, info in metadata.items():
        if file_id in existing_insights:
            continue  # 已处理

        # 检查可用文本
        text_info = {'file_id': file_id}

        # 检查清洗后文件
        if (info.get("cleaning_status") == "cleaned" and
                "cleaned_text_path" in info and
                os.path.exists(info["cleaned_text_path"])):
            try:
                with open(info["cleaned_text_path"], 'r', encoding='utf-8') as f:
                    cleaned_text = f.read()
                if len(cleaned_text.strip()) >= 1000:
                    text_info['text_path'] = info["cleaned_text_path"]
                    text_info['source_type'] = "cleaned"
                    text_info['text_length'] = len(cleaned_text)
                    missing_docs[file_id] = text_info
                    continue
            except:
                pass

        # 检查原始文件
        if (info.get("status") == "text_extracted" and
                "text_path" in info and
                os.path.exists(info["text_path"])):
            try:
                with open(info["text_path"], 'r', encoding='utf-8') as f:
                    basic_text = f.read()
                if len(basic_text.strip()) >= 500:
                    text_info['text_path'] = info["text_path"]
                    text_info['source_type'] = "basic"
                    text_info['text_length'] = len(basic_text)
                    missing_docs[file_id] = text_info
            except:
                pass

    print(f"需补充处理: {len(missing_docs)} 个文档")
    return missing_docs, existing_insights


def supplement_missing_insights():
    """补充缺失的洞察（调试版本）"""

    missing_docs, existing_insights = find_missing_documents()

    if not missing_docs:
        print("所有文档都已有结构化洞察！")
        return

    extractor = SupplementExtractor()
    stats = {"experimental_paper": 0, "theoretical_paper": 0, "patent": 0, "unknown": 0, "failed": 0}

    print(f"\n开始补充处理 {len(missing_docs)} 个文档...")
    print("详细调试模式已启用")
    print("-" * 60)

    for i, (file_id, file_info) in enumerate(missing_docs.items()):
        print(f"\n{'=' * 20} 处理文档 {i + 1}/{len(missing_docs)} {'=' * 20}")
        print(f"文档ID: {file_id}")
        print(f"数据源: {file_info['source_type']}")
        print(f"文件路径: {file_info['text_path']}")
        print(f"文本长度: {file_info['text_length']} 字符")
        print("-" * 60)

        try:
            # 读取文本
            with open(file_info['text_path'], 'r', encoding='utf-8') as f:
                document_text = f.read()

            # 提取洞察
            insights = extractor.extract_insights(
                document_text,
                file_id,
                file_info['source_type']
            )

            if insights:
                # 添加到已有洞察中
                existing_insights[file_id] = insights

                # 保存单独文件
                extractor.save_individual_insight(file_id, insights)

                extractor.success_count += 1
                doc_type = insights.get('extraction_metadata', {}).get('detected_doc_type', 'unknown')
                stats[doc_type] += 1
                print(f"文档处理成功!")

            else:
                stats["failed"] += 1
                print(f"文档处理失败!")

            extractor.total_count += 1

            # 每处理一个保存一次（调试模式）
            with open(STRUCTURAL_DATA_PATH, 'w', encoding='utf-8') as f:
                json.dump(existing_insights, f, indent=2, ensure_ascii=False)
            print(f"已保存进度... 已处理 {extractor.total_count}/{len(missing_docs)}")

            # 增加API间隔用于调试
            if i < len(missing_docs) - 1:  # 最后一个不等待
                print(f"等待3秒后处理下一个文档...")
                time.sleep(3)

        except Exception as e:
            stats["failed"] += 1
            print(f"处理异常: {file_id}, {e}")
            import traceback
            print("完整错误堆栈:")
            print(traceback.format_exc())
            continue

    # 最终保存
    with open(STRUCTURAL_DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(existing_insights, f, indent=2, ensure_ascii=False)

    # 显示结果
    print("\n" + "=" * 60)
    print("补充提取完成！")
    print("=" * 60)
    print(f"处理统计:")
    print(f"  实验性论文: {stats['experimental_paper']} 个")
    print(f"  理论性论文: {stats['theoretical_paper']} 个")
    print(f"  专利文档: {stats['patent']} 个")
    print(f"  未知类型: {stats['unknown']} 个")
    print(f"  处理失败: {stats['failed']} 个")

    if extractor.total_count > 0:
        success_rate = extractor.success_count / extractor.total_count * 100
        print(f"  成功率: {extractor.success_count}/{extractor.total_count} ({success_rate:.1f}%)")

    print(f"\n总洞察数量: {len(existing_insights)} 个")
    print(f"结果保存在: {STRUCTURAL_DATA_PATH}")


def main():
    """主函数"""
    supplement_missing_insights()


if __name__ == "__main__":
    main()