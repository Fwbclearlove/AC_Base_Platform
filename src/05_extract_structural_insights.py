# src/05_extract_structural_insights_safe.py
import os
import json
import time
import re
from zhipuai import ZhipuAI
# src/05_extract_structural_insights_safe.py
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

# 输出路径
STRUCTURAL_DIR = PROCESSED_DATA_DIR / "structural_insights"
STRUCTURAL_DIR.mkdir(exist_ok=True)
STRUCTURAL_DATA_PATH = STRUCTURAL_DIR / "structural_insights.json"
INDIVIDUAL_INSIGHTS_DIR = STRUCTURAL_DIR / "individual"
INDIVIDUAL_INSIGHTS_DIR.mkdir(exist_ok=True)

# 文档类型检测的关键词
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

# 严格约束的基础提示词模板
BASE_EXTRACTION_PROMPT = """
你是一位极其严格的学术文献分析专家。你的任务是从文档中提取结构化信息，但必须遵循铁律：

【铁律】：
1. 绝对不能编造、推测、补充任何原文没有的信息
2. 如果信息不存在，必须填写"原文无此信息"
3. 如果信息模糊不清，必须填写"原文未明确提及"
4. 每个提取的信息都必须在原文中有确切的文字依据
5. 宁可留空或标注不存在，也不能猜测

【重要提示】：
- 只输出一个完整的JSON对象
- JSON后不要添加任何解释文字
- 确保所有括号和引号正确匹配
- 不要使用markdown代码块标记

文档原文：
---
{document_text}
---

{specific_instructions}

请严格按照JSON格式输出，对于不存在的信息必须如实标注。只输出JSON，不要任何额外文字：
"""

# 实验性论文的具体指令
EXPERIMENTAL_PAPER_INSTRUCTIONS = """
这是一篇包含实验的论文。请提取以下信息，但只能提取原文明确存在的：

{{
  "document_metadata": {{
    "document_type": "experimental_paper",
    "title": "原文确切标题或'原文无此信息'",
    "authors": ["原文明确列出的作者，如无则为空数组"],
    "institutions": ["原文明确提到的机构，如无则为空数组"],
    "publication_venue": "原文中的发表venue或'原文无此信息'"
  }},
  "technical_relationships": {{
    "base_methods": [
      {{
        "method_name": "原文明确提到的基础方法名或'原文无此信息'",
        "relationship_type": "基于|改进|扩展|结合 或'原文无此信息'",
        "evidence_text": "原文确切句子或'原文无此信息'"
      }}
    ],
    "compared_methods": [
      {{
        "method_name": "原文明确对比的方法名或'原文无此信息'",
        "comparison_result": "原文明确的对比结果或'原文无此信息'",
        "evidence_text": "原文确切句子或'原文无此信息'"
      }}
    ]
  }},
  "experimental_setup": {{
    "datasets_used": [
      {{
        "dataset_name": "原文明确提到的数据集名或'原文无此信息'",
        "dataset_description": "原文对数据集的描述或'原文无此信息'",
        "evidence_text": "原文确切句子或'原文无此信息'"
      }}
    ],
    "evaluation_metrics": ["原文明确提到的评估指标，如无则为空数组"],
    "baseline_methods": ["原文明确的基准方法，如无则为空数组"]
  }},
  "performance_results": {{
    "quantitative_results": [
      {{
        "metric_name": "原文明确的指标名或'原文无此信息'",
        "our_result": "原文中我们方法的数值或'原文无此信息'",
        "baseline_result": "原文中基准的数值或'原文无此信息'",
        "dataset": "对应的数据集或'原文无此信息'",
        "evidence_text": "原文确切句子或'原文无此信息'"
      }}
    ]
  }},
  "innovation_analysis": {{
    "stated_contributions": ["原文明确声明的贡献，如无则为空数组"],
    "stated_novelty": ["原文明确声明的新颖性，如无则为空数组"],
    "stated_advantages": ["原文明确声明的优势，如无则为空数组"]
  }},
  "limitations": {{
    "acknowledged_limitations": ["原文明确承认的局限性，如无则为空数组"],
    "failure_cases": ["原文提到的失败案例，如无则为空数组"]
  }}
}}
"""

# 理论性论文的具体指令
THEORETICAL_PAPER_INSTRUCTIONS = """
这是一篇理论性论文。请提取以下信息，不要尝试提取不存在的实验信息：

{{
  "document_metadata": {{
    "document_type": "theoretical_paper",
    "title": "原文确切标题或'原文无此信息'",
    "authors": ["原文明确列出的作者，如无则为空数组"],
    "institutions": ["原文明确提到的机构，如无则为空数组"]
  }},
  "theoretical_contributions": {{
    "main_theoretical_results": ["原文明确的理论结果，如无则为空数组"],
    "theorems_proposed": ["原文提出的定理，如无则为空数组"],
    "mathematical_models": ["原文的数学模型，如无则为空数组"]
  }},
  "technical_relationships": {{
    "builds_upon": ["原文明确基于的理论或方法，如无则为空数组"],
    "extends": ["原文明确扩展的理论，如无则为空数组"],
    "relates_to": ["原文明确相关的理论，如无则为空数组"]
  }},
  "innovation_analysis": {{
    "theoretical_novelty": ["原文声明的理论新颖性，如无则为空数组"],
    "mathematical_contributions": ["原文声明的数学贡献，如无则为空数组"]
  }},
  "applications": {{
    "potential_applications": ["原文提到的潜在应用，如无则为空数组"],
    "application_domains": ["原文提到的应用领域，如无则为空数组"]
  }},
  "note": "理论性论文，不包含实验数据"
}}
"""

# 专利文档的具体指令
PATENT_INSTRUCTIONS = """
这是一篇专利文档。请提取以下信息：

{{
  "document_metadata": {{
    "document_type": "patent",
    "title": "专利确切名称或'原文无此信息'",
    "inventors": ["原文明确列出的发明人，如无则为空数组"],
    "applicant": "申请人信息或'原文无此信息'",
    "patent_number": "专利号或'原文无此信息'"
  }},
  "technical_solution": {{
    "technical_problem": "原文描述的技术问题或'原文无此信息'",
    "solution_overview": "原文的技术方案概述或'原文无此信息'",
    "key_technical_features": ["原文明确的关键技术特征，如无则为空数组"]
  }},
  "implementation": {{
    "embodiments": ["原文描述的实施例，如无则为空数组"],
    "technical_effects": ["原文声称的技术效果，如无则为空数组"]
  }},
  "application_scope": {{
    "application_fields": ["原文明确的应用领域，如无则为空数组"],
    "use_scenarios": ["原文描述的使用场景，如无则为空数组"]
  }},
  "claims_info": {{
    "main_claims": ["权利要求的核心内容，如无则为空数组"]
  }}
}}
"""

# 保守提取的指令（用于未知类型文档）
CONSERVATIVE_INSTRUCTIONS = """
文档类型不明确，进行保守提取：

{{
  "document_metadata": {{
    "document_type": "unknown",
    "title": "原文确切标题或'原文无此信息'",
    "authors_or_creators": ["原文明确的作者或创作者，如无则为空数组"]
  }},
  "main_content": {{
    "stated_purpose": "原文明确的目的或'原文无此信息'",
    "key_concepts": ["原文明确的关键概念，如无则为空数组"],
    "main_methods": ["原文明确的方法，如无则为空数组"]
  }},
  "relationships": {{
    "references_to": ["原文明确引用或提及的其他工作，如无则为空数组"],
    "builds_on": ["原文明确基于的工作，如无则为空数组"]
  }},
  "note": "文档类型不明确，仅提取最基础信息"
}}
"""


def safe_json_parse(json_string, file_id):
    """安全的JSON解析，处理各种格式问题"""

    # 第一步：基础清理
    cleaned = json_string.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    # 第二步：尝试直接解析
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"直接解析失败: {file_id}, 错误: {e}")

        # 第三步：尝试提取第一个完整的JSON对象
        try:
            # 找到第一个 { 的位置
            start_pos = cleaned.find('{')
            if start_pos == -1:
                print(f"未找到JSON开始标记: {file_id}")
                return None

            # 从第一个 { 开始，找到匹配的 }
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

            # 提取第一个完整的JSON
            first_json = cleaned[start_pos:end_pos]
            return json.loads(first_json)

        except json.JSONDecodeError as e2:
            print(f"提取第一个JSON失败: {file_id}, 错误: {e2}")

            # 第四步：显示原始返回内容用于调试
            print(f"原始返回内容预览 ({file_id}):")
            print("=" * 50)
            print(cleaned[:800] + "..." if len(cleaned) > 800 else cleaned)
            print("=" * 50)

            return None


class SafeStructuralExtractor:
    """安全的结构化信息提取器"""

    def __init__(self):
        self.max_text_length = 15000
        self.retry_count = 3
        self.success_count = 0
        self.total_count = 0

    def detect_document_type(self, text):
        """检测文档类型"""
        text_lower = text.lower()

        # 计算不同类型的关键词得分
        experimental_score = sum(1 for kw in EXPERIMENTAL_KEYWORDS if kw in text_lower)
        theoretical_score = sum(1 for kw in THEORETICAL_KEYWORDS if kw in text_lower)
        patent_score = sum(1 for kw in PATENT_KEYWORDS if kw in text_lower)

        print(f"文档类型评分 - 实验性: {experimental_score}, 理论性: {theoretical_score}, 专利: {patent_score}")

        if patent_score >= 3:
            return "patent"
        elif experimental_score >= 5:
            return "experimental_paper"
        elif theoretical_score >= 3 and experimental_score < 3:
            return "theoretical_paper"
        else:
            return "unknown"

    def truncate_text_smartly(self, text, doc_type):
        """根据文档类型智能截取文本"""
        if len(text) <= self.max_text_length:
            return text

        paragraphs = text.split('\n\n')
        important_sections = []
        remaining_sections = []

        # 根据文档类型选择重要关键词
        if doc_type == "experimental_paper":
            important_keywords = [
                'abstract', 'experiment', 'evaluation', 'result', 'dataset', 'performance',
                'method', 'approach', 'comparison', 'baseline',
                '摘要', '实验', '评估', '结果', '数据集', '性能', '方法', '对比'
            ]
        elif doc_type == "theoretical_paper":
            important_keywords = [
                'abstract', 'theorem', 'proof', 'proposition', 'definition', 'analysis',
                'theory', 'mathematical', 'formal',
                '摘要', '定理', '证明', '命题', '定义', '分析', '理论', '数学'
            ]
        elif doc_type == "patent":
            important_keywords = [
                '技术领域', '背景技术', '发明内容', '技术方案', '实施例',
                'technical field', 'background', 'invention', 'claims', 'embodiment'
            ]
        else:
            important_keywords = [
                'abstract', 'introduction', 'method', 'conclusion',
                '摘要', '引言', '方法', '结论'
            ]

        for para in paragraphs:
            para_lower = para.lower()
            if any(keyword in para_lower for keyword in important_keywords):
                important_sections.append(para)
            else:
                remaining_sections.append(para)

        # 组装文本
        result_text = '\n\n'.join(important_sections)

        # 如果还有空间，添加其他部分
        for para in remaining_sections:
            if len(result_text + '\n\n' + para) <= self.max_text_length:
                result_text += '\n\n' + para
            else:
                break

        return result_text

    def get_extraction_instructions(self, doc_type):
        """根据文档类型获取提取指令"""
        if doc_type == "experimental_paper":
            return EXPERIMENTAL_PAPER_INSTRUCTIONS
        elif doc_type == "theoretical_paper":
            return THEORETICAL_PAPER_INSTRUCTIONS
        elif doc_type == "patent":
            return PATENT_INSTRUCTIONS
        else:
            return CONSERVATIVE_INSTRUCTIONS

    def validate_extraction_quality(self, insights, original_text, doc_type):
        """验证提取质量，检测可能的编造信息"""
        warnings = []
        suspicious_count = 0

        def check_text_existence(field_name, value, original_text):
            """检查字段值是否在原文中存在"""
            if not value or value in ["原文无此信息", "原文未明确提及", ""]:
                return True  # 诚实标注，没问题

            if isinstance(value, str):
                # 检查字符串是否在原文中（允许部分匹配）
                if len(value) > 3 and value not in original_text:
                    return False
            elif isinstance(value, list):
                # 检查列表中的每个元素
                for item in value:
                    if isinstance(item, str) and len(item) > 3:
                        if item not in original_text and item not in ["原文无此信息", "原文未明确提及"]:
                            return False
            return True

        # 检查基本元数据
        metadata = insights.get('document_metadata', {})
        for field, value in metadata.items():
            if not check_text_existence(field, value, original_text):
                warnings.append(f"可疑{field}: {value}")
                suspicious_count += 1

        # 检查技术关系
        tech_relations = insights.get('technical_relationships', {})
        for method_list_name in ['base_methods', 'compared_methods']:
            for method in tech_relations.get(method_list_name, []):
                if isinstance(method, dict):
                    for field, value in method.items():
                        if not check_text_existence(field, value, original_text):
                            warnings.append(f"可疑技术关系{field}: {value}")
                            suspicious_count += 1

        # 特定于实验性论文的检查
        if doc_type == "experimental_paper":
            # 检查数据集名称
            exp_setup = insights.get('experimental_setup', {})
            for dataset in exp_setup.get('datasets_used', []):
                if isinstance(dataset, dict):
                    dataset_name = dataset.get('dataset_name', '')
                    if not check_text_existence('dataset_name', dataset_name, original_text):
                        warnings.append(f"可疑数据集: {dataset_name}")
                        suspicious_count += 1

            # 检查性能数据
            performance = insights.get('performance_results', {})
            for result in performance.get('quantitative_results', []):
                if isinstance(result, dict):
                    for field, value in result.items():
                        if field in ['our_result', 'baseline_result'] and value != "原文无此信息":
                            # 检查数值是否合理且在原文中存在
                            if not check_text_existence(field, value, original_text):
                                warnings.append(f"可疑性能数据{field}: {value}")
                                suspicious_count += 1

        # 计算验证得分
        total_checks = len(warnings) + 10  # 假设进行了约10个检查
        validation_score = max(0, 100 - (suspicious_count * 20))  # 每个可疑项扣20分

        return {
            'warnings': warnings,
            'suspicious_count': suspicious_count,
            'validation_score': validation_score,
            'quality_level': 'high' if validation_score >= 80 else 'medium' if validation_score >= 60 else 'low'
        }

    def extract_structural_insights(self, text, file_id, source_type):
        """安全的结构化洞察提取"""

        # 检测文档类型
        doc_type = self.detect_document_type(text)
        print(f"检测文档类型: {file_id} -> {doc_type} (来源: {source_type})")

        # 智能截取文本
        processed_text = self.truncate_text_smartly(text, doc_type)

        # 获取对应的提取指令
        extraction_instructions = self.get_extraction_instructions(doc_type)

        # 构建完整提示词
        full_prompt = BASE_EXTRACTION_PROMPT.format(
            document_text=processed_text,
            specific_instructions=extraction_instructions
        )

        for attempt in range(self.retry_count):
            try:
                response = CLIENT.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {
                            "role": "user",
                            "content": full_prompt
                        }
                    ],
                    temperature=0.01,  # 更低温度确保严格按原文
                    max_tokens=1800,  # 稍微减少token数
                    stop=["\n\n注意", "\n\n说明", "```"],  # 添加停止词
                )

                result_str = response.choices[0].message.content.strip()

                # 使用安全JSON解析
                insights_json = safe_json_parse(result_str, file_id)

                if insights_json is None:
                    print(f"JSON解析失败: {file_id}, 尝试 {attempt + 1}")
                    continue

                # 验证提取质量
                validation_result = self.validate_extraction_quality(insights_json, text, doc_type)

                # 添加验证信息
                insights_json['extraction_metadata'] = {
                    'file_id': file_id,
                    'detected_doc_type': doc_type,
                    'source_type': source_type,  # 标记数据来源
                    'extraction_time': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'validation': validation_result,
                    'text_length': len(text),
                    'processed_text_length': len(processed_text)
                }

                # 检查验证质量
                if validation_result['quality_level'] == 'low':
                    print(f"警告: {file_id} 提取质量较低 (得分: {validation_result['validation_score']})")
                    print(f"可疑项: {validation_result['suspicious_count']} 个")
                    if attempt < self.retry_count - 1:
                        print(f"重试提取...")
                        time.sleep(3)
                        continue

                print(
                    f"成功提取结构化洞察: {file_id} ({doc_type}, {source_type}) - 质量: {validation_result['quality_level']}")
                return insights_json

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

    def save_individual_insights(self, file_id, insights_data):
        """保存单个文档的结构化洞察"""
        doc_type = insights_data.get('extraction_metadata', {}).get('detected_doc_type', 'unknown')
        source_type = insights_data.get('extraction_metadata', {}).get('source_type', 'unknown')
        insights_file = INDIVIDUAL_INSIGHTS_DIR / f"{file_id}_{doc_type}_{source_type}_structural.json"
        with open(insights_file, 'w', encoding='utf-8') as f:
            json.dump(insights_data, f, indent=2, ensure_ascii=False)


def extract_all_structural_insights_safe():
    """安全地提取所有文档的结构化洞察（支持回退策略）"""
    print("=" * 60)
    print("安全结构化洞察提取系统（智能回退版本）")
    print("清洗后文本过短时自动回退到原始提取文本")
    print("=" * 60)

    metadata_path = PROCESSED_DATA_DIR / "metadata.json"

    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print("错误：metadata.json文件未找到")
        return

    extractor = SafeStructuralExtractor()

    # 收集所有可用的文件（清洗后的和原始提取的）
    print("收集可用文件...")
    available_files = {}

    for file_id, info in metadata.items():
        file_info = {'file_id': file_id}

        # 检查清洗后的文件
        if (info.get("cleaning_status") == "cleaned" and
                "cleaned_text_path" in info and
                os.path.exists(info["cleaned_text_path"])):
            file_info['cleaned_path'] = info["cleaned_text_path"]

        # 检查原始提取的文件
        if (info.get("status") == "text_extracted" and
                "text_path" in info and
                os.path.exists(info["text_path"])):
            file_info['basic_path'] = info["text_path"]

        # 只有至少有一种文件可用才加入处理列表
        if 'cleaned_path' in file_info or 'basic_path' in file_info:
            available_files[file_id] = file_info

    if not available_files:
        print("没有找到可处理的文件")
        return

    print(f"共找到 {len(available_files)} 个可处理的文档")

    # 加载已有的结构化洞察
    all_insights = {}
    if os.path.exists(STRUCTURAL_DATA_PATH):
        try:
            with open(STRUCTURAL_DATA_PATH, 'r', encoding='utf-8') as f:
                all_insights = json.load(f)
            print(f"加载了 {len(all_insights)} 个已有洞察")
        except:
            print("洞察文件损坏，将创建新的")
            all_insights = {}

    # 统计信息
    stats = {
        "experimental_paper": 0,
        "theoretical_paper": 0,
        "patent": 0,
        "unknown": 0,
        "failed": 0,
        "quality_high": 0,
        "quality_medium": 0,
        "quality_low": 0,
        "used_cleaned": 0,
        "used_basic": 0,
        "fallback_cases": 0
    }

    # 开始提取结构化洞察
    for file_id, file_info in tqdm(available_files.items(), desc="安全提取结构化洞察"):
        # 跳过已处理的文件
        if file_id in all_insights:
            print(f"跳过已有洞察: {file_id}")
            continue

        document_text = None
        source_type = None
        text_path = None

        # 智能选择文本来源
        try:
            # 优先尝试使用清洗后的文件
            if 'cleaned_path' in file_info:
                with open(file_info['cleaned_path'], 'r', encoding='utf-8') as f:
                    cleaned_text = f.read()

                if len(cleaned_text.strip()) >= 1000:  # 清洗后文本足够长
                    document_text = cleaned_text
                    source_type = "cleaned"
                    text_path = file_info['cleaned_path']
                    stats["used_cleaned"] += 1
                    print(f"使用清洗后文本: {file_id} (长度: {len(cleaned_text)})")
                else:
                    print(f"清洗后文本过短 ({len(cleaned_text)} 字符): {file_id}, 尝试回退到原始文本")
                    # 回退到原始提取文件
                    if 'basic_path' in file_info:
                        with open(file_info['basic_path'], 'r', encoding='utf-8') as f:
                            basic_text = f.read()

                        if len(basic_text.strip()) >= 500:  # 原始文本的最低要求更低
                            document_text = basic_text
                            source_type = "basic_fallback"
                            text_path = file_info['basic_path']
                            stats["used_basic"] += 1
                            stats["fallback_cases"] += 1
                            print(f"回退使用原始文本: {file_id} (长度: {len(basic_text)})")
                        else:
                            print(f"原始文本也过短 ({len(basic_text)} 字符): {file_id}, 跳过")
                            continue
                    else:
                        print(f"没有可用的原始文本: {file_id}, 跳过")
                        continue

            # 如果没有清洗后文件，直接使用原始文件
            elif 'basic_path' in file_info:
                with open(file_info['basic_path'], 'r', encoding='utf-8') as f:
                    basic_text = f.read()

                if len(basic_text.strip()) >= 500:
                    document_text = basic_text
                    source_type = "basic_only"
                    text_path = file_info['basic_path']
                    stats["used_basic"] += 1
                    print(f"使用原始文本（无清洗版本）: {file_id} (长度: {len(basic_text)})")
                else:
                    print(f"原始文本过短 ({len(basic_text)} 字符): {file_id}, 跳过")
                    continue

            else:
                print(f"没有可用文本文件: {file_id}, 跳过")
                continue

            # 提取结构化洞察
            insights = extractor.extract_structural_insights(document_text, file_id, source_type)

            if insights:
                # 保存到总集合
                all_insights[file_id] = insights

                # 保存单独文件
                extractor.save_individual_insights(file_id, insights)

                extractor.success_count += 1

                # 统计文档类型和质量
                doc_type = insights.get('extraction_metadata', {}).get('detected_doc_type', 'unknown')
                quality_level = insights.get('extraction_metadata', {}).get('validation', {}).get('quality_level',
                                                                                                  'unknown')

                stats[doc_type] = stats.get(doc_type, 0) + 1
                stats[f"quality_{quality_level}"] = stats.get(f"quality_{quality_level}", 0) + 1

                # 显示洞察预览
                print(f"{file_id} ({doc_type}, {source_type}) 结构化洞察预览:")
                print(f"   质量等级: {quality_level}")

                validation = insights.get('extraction_metadata', {}).get('validation', {})
                if validation.get('warnings'):
                    print(f"   验证警告: {len(validation['warnings'])} 个")
                else:
                    print(f"   验证通过: 无可疑信息")
                print()
            else:
                stats["failed"] += 1
                print(f"提取失败: {file_id}")

            extractor.total_count += 1

            # 每处理3个文件保存一次
            if extractor.total_count % 3 == 0:
                with open(STRUCTURAL_DATA_PATH, 'w', encoding='utf-8') as f:
                    json.dump(all_insights, f, indent=2, ensure_ascii=False)
                print(f"中间保存... 已处理 {extractor.total_count}")

            # API调用间隔
            time.sleep(4)

        except Exception as e:
            stats["failed"] += 1
            print(f"处理异常: {file_id}, {e}")
            continue

    # 最终保存
    with open(STRUCTURAL_DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_insights, f, indent=2, ensure_ascii=False)

    # 生成质量报告
    generate_quality_report(all_insights, stats)

    # 显示统计结果
    print(f"\n安全结构化洞察提取完成！")
    print(f"处理统计:")
    print(f"   实验性论文: {stats.get('experimental_paper', 0)} 个")
    print(f"   理论性论文: {stats.get('theoretical_paper', 0)} 个")
    print(f"   专利文档: {stats.get('patent', 0)} 个")
    print(f"   未知类型: {stats.get('unknown', 0)} 个")
    print(f"   处理失败: {stats.get('failed', 0)} 个")
    print(f"\n数据来源分布:")
    print(f"   使用清洗后文本: {stats.get('used_cleaned', 0)} 个")
    print(f"   使用原始文本: {stats.get('used_basic', 0)} 个")
    print(f"   回退案例: {stats.get('fallback_cases', 0)} 个")
    print(f"\n质量分布:")
    print(f"   高质量: {stats.get('quality_high', 0)} 个")
    print(f"   中等质量: {stats.get('quality_medium', 0)} 个")
    print(f"   低质量: {stats.get('quality_low', 0)} 个")

    if extractor.total_count > 0:
        print(
            f"   总成功率: {extractor.success_count}/{extractor.total_count} ({extractor.success_count / extractor.total_count * 100:.1f}%)")

    print(f"结构化洞察保存在: {STRUCTURAL_DATA_PATH}")
    print(f"单独文件保存在: {INDIVIDUAL_INSIGHTS_DIR}")

    return all_insights


def generate_quality_report(insights_data, stats):
    """生成质量报告"""
    quality_report = {
        'generation_time': time.strftime("%Y-%m-%d %H:%M:%S"),
        'total_documents': len(insights_data),
        'document_type_distribution': {
            'experimental_paper': stats.get('experimental_paper', 0),
            'theoretical_paper': stats.get('theoretical_paper', 0),
            'patent': stats.get('patent', 0),
            'unknown': stats.get('unknown', 0)
        },
        'quality_distribution': {
            'high': stats.get('quality_high', 0),
            'medium': stats.get('quality_medium', 0),
            'low': stats.get('quality_low', 0)
        },
        'source_distribution': {
            'cleaned_text': stats.get('used_cleaned', 0),
            'basic_text': stats.get('used_basic', 0),
            'fallback_cases': stats.get('fallback_cases', 0)
        },
        'high_quality_documents': [],
        'low_quality_documents': [],
        'fallback_documents': []
    }

    # 识别高质量、低质量和回退文档
    for file_id, insights in insights_data.items():
        validation = insights.get('extraction_metadata', {}).get('validation', {})
        quality_level = validation.get('quality_level', 'unknown')
        source_type = insights.get('extraction_metadata', {}).get('source_type', 'unknown')

        if quality_level == 'high':
            quality_report['high_quality_documents'].append({
                'file_id': file_id,
                'doc_type': insights.get('extraction_metadata', {}).get('detected_doc_type'),
                'source_type': source_type,
                'validation_score': validation.get('validation_score', 0)
            })
        elif quality_level == 'low':
            quality_report['low_quality_documents'].append({
                'file_id': file_id,
                'doc_type': insights.get('extraction_metadata', {}).get('detected_doc_type'),
                'source_type': source_type,
                'validation_score': validation.get('validation_score', 0),
                'warnings_count': len(validation.get('warnings', []))
            })

        if 'fallback' in source_type:
            quality_report['fallback_documents'].append({
                'file_id': file_id,
                'doc_type': insights.get('extraction_metadata', {}).get('detected_doc_type'),
                'source_type': source_type
            })

    # 保存质量报告
    quality_report_path = STRUCTURAL_DIR / "quality_report.json"
    with open(quality_report_path, 'w', encoding='utf-8') as f:
        json.dump(quality_report, f, indent=2, ensure_ascii=False)

    print(f"质量报告保存在: {quality_report_path}")


def main():
    """主函数"""
    insights_data = extract_all_structural_insights_safe()
    return insights_data


if __name__ == "__main__":
    main()