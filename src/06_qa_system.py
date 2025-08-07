# src/06_qa_system.py
import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
from zhipuai import ZhipuAI
from collections import defaultdict

from config import PROCESSED_DATA_DIR, VERBOSE

# 修改导入语句 - 使用动态导入处理数字开头的模块名
import importlib
import sys

# 确保当前目录在Python路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 动态导入05_retrieval_system
try:
    retrieval_module = importlib.import_module('05_retrieval_system')
    SearchEngine = retrieval_module.SearchEngine
    SearchQuery = retrieval_module.SearchQuery
    SearchResult = retrieval_module.SearchResult
except ImportError as e:
    print(f"导入错误: {e}")
    # 尝试另一种方式
    try:
        exec(open('05_retrieval_system.py').read())
    except Exception as e2:
        print(f"无法加载检索系统: {e2}")
        raise

# 设置日志
logging.basicConfig(level=logging.INFO if VERBOSE else logging.WARNING)
logger = logging.getLogger(__name__)

# LLM配置
LLM_API_KEY = "838cc9e6876a4fea971b3728af105b56.1KDgfLzNHnfllnhb"  # 需要替换为实际的API Key
LLM_MODEL = "glm-4"
LLM_CLIENT = None

# 系统提示词
SYSTEM_PROMPT = """你是一个专业的学术研究助手，专门帮助用户理解和探索学术知识。

你的职责包括：
1. 基于检索到的学术文献内容，准确回答用户问题
2. 保持学术严谨性，不编造不存在的信息
3. 如果检索结果不足以回答问题，诚实告知用户
4. 适当引用来源，让用户知道信息的出处
5. 根据用户背景（如低年级学生）调整回答的深度和专业程度

回答原则：
- 准确性：只使用检索到的内容，不臆测
- 可读性：用清晰易懂的语言解释复杂概念
- 完整性：尽可能全面地回答问题
- 学术性：保持专业和客观的语气
"""


@dataclass
class QAContext:
    """问答上下文"""
    query: str
    search_results: List[Dict]
    conversation_history: List[Dict]
    user_profile: Optional[Dict] = None


@dataclass
class QAResponse:
    """问答响应"""
    answer: str
    citations: List[Dict]
    confidence: float
    search_quality: str
    suggestions: List[str]


class AcademicQASystem:
    """学术问答系统"""

    def __init__(self, search_engine: Optional[SearchEngine] = None):
        self.search_engine = search_engine or SearchEngine()
        self.llm_client = None
        self.conversation_memory = defaultdict(list)
        self._initialize_llm()

    def _initialize_llm(self):
        """初始化LLM客户端"""
        global LLM_CLIENT
        if not LLM_CLIENT:
            LLM_CLIENT = ZhipuAI(api_key=LLM_API_KEY)
        self.llm_client = LLM_CLIENT
        logger.info("LLM客户端初始化完成")

    def answer_question(
            self,
            question: str,
            conversation_id: str = "default",
            search_type: str = "hybrid",
            top_k: int = 5,
            user_level: str = "general"  # student/researcher/general
    ) -> QAResponse:
        """回答用户问题"""

        logger.info(f"处理问题: {question[:50]}...")

        # 1. 执行检索
        search_results = self._search_relevant_content(
            question, search_type, top_k
        )

        # 2. 评估检索质量
        search_quality = self._evaluate_search_quality(search_results)

        # 3. 构建问答上下文
        qa_context = QAContext(
            query=question,
            search_results=search_results,
            conversation_history=self.conversation_memory[conversation_id][-5:],  # 最近5轮
            user_profile={"level": user_level}
        )

        # 4. 生成答案
        answer, confidence = self._generate_answer(qa_context)

        # 5. 提取引用信息
        citations = self._extract_citations(search_results, answer)

        # 6. 生成后续建议
        suggestions = self._generate_suggestions(question, answer, search_results)

        # 7. 更新对话历史
        self._update_conversation_history(
            conversation_id, question, answer
        )

        return QAResponse(
            answer=answer,
            citations=citations,
            confidence=confidence,
            search_quality=search_quality,
            suggestions=suggestions
        )

    def _search_relevant_content(
            self,
            question: str,
            search_type: str,
            top_k: int
    ) -> List[Dict]:
        """搜索相关内容"""

        # 初始化搜索引擎
        if not self.search_engine.is_loaded:
            self.search_engine.initialize()

        # 执行搜索
        results = self.search_engine.search(
            query_text=question,
            search_type=search_type,
            top_k=top_k
        )

        return results

    def _evaluate_search_quality(self, search_results: List[Dict]) -> str:
        """评估检索质量"""
        if not search_results:
            return "no_results"

        # 基于得分评估
        top_score = search_results[0]['score'] if search_results else 0
        avg_score = sum(r['score'] for r in search_results) / len(search_results)

        if top_score > 0.8 and avg_score > 0.6:
            return "excellent"
        elif top_score > 0.6 and avg_score > 0.4:
            return "good"
        elif top_score > 0.4:
            return "fair"
        else:
            return "poor"

    def _generate_answer(self, context: QAContext) -> Tuple[str, float]:
        """生成答案"""

        # 构建提示词
        prompt = self._build_prompt(context)

        # 调用LLM
        try:
            response = self.llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # 低温度保证稳定性
                max_tokens=1500,
            )

            answer = response.choices[0].message.content.strip()

            # 计算置信度
            confidence = self._calculate_confidence(
                answer, context.search_results
            )

            return answer, confidence

        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            return self._generate_fallback_answer(context), 0.3

    def _build_prompt(self, context: QAContext) -> str:
        """构建LLM提示词"""

        # 用户级别说明
        level_instructions = {
            "student": "请用通俗易懂的语言解释，避免过多专业术语，适合低年级学生理解。",
            "researcher": "请提供专业深入的分析，包括技术细节和理论基础。",
            "general": "请用清晰准确的语言回答，平衡专业性和可读性。"
        }

        user_level = context.user_profile.get("level", "general")
        level_instruction = level_instructions.get(user_level, level_instructions["general"])

        # 构建检索内容
        retrieved_content = self._format_search_results(context.search_results)

        # 构建对话历史
        history_text = self._format_conversation_history(context.conversation_history)

        prompt = f"""基于以下检索到的学术内容，请回答用户的问题。

{level_instruction}

用户问题：{context.query}

检索到的相关内容：
{retrieved_content}

{history_text}

请注意：
1. 只基于提供的内容回答，不要编造信息
2. 如果内容不足以完全回答问题，请明确指出
3. 适当引用来源（如"根据文档X..."）
4. 保持学术严谨性

请回答用户的问题："""

        return prompt

    def _format_search_results(self, results: List[Dict]) -> str:
        """格式化检索结果"""
        formatted_results = []

        for i, result in enumerate(results, 1):
            source = result['source_id']
            section = result['section']
            text = result['text']
            score = result['score']

            # 截断过长文本
            if len(text) > 500:
                text = text[:500] + "..."

            formatted_results.append(
                f"【来源{i}】{source} - {section} (相关度: {score:.2f})\n{text}\n"
            )

        return "\n".join(formatted_results)

    def _format_conversation_history(self, history: List[Dict]) -> str:
        """格式化对话历史"""
        if not history:
            return ""

        history_text = "对话历史：\n"
        for turn in history[-3:]:  # 只保留最近3轮
            history_text += f"用户：{turn['question']}\n"
            history_text += f"助手：{turn['answer'][:100]}...\n\n"

        return history_text

    def _calculate_confidence(self, answer: str, search_results: List[Dict]) -> float:
        """计算答案置信度"""
        # 基于多个因素计算置信度
        confidence = 0.5  # 基础置信度

        # 1. 检索结果质量
        if search_results:
            avg_score = sum(r['score'] for r in search_results) / len(search_results)
            confidence += avg_score * 0.3

        # 2. 答案长度（太短或太长都降低置信度）
        answer_length = len(answer)
        if 100 < answer_length < 800:
            confidence += 0.1

        # 3. 是否包含"不确定"等词汇
        uncertain_words = ['不确定', '可能', '也许', '不清楚', '信息不足']
        if any(word in answer for word in uncertain_words):
            confidence -= 0.2

        return max(0.1, min(1.0, confidence))

    def _extract_citations(self, search_results: List[Dict], answer: str) -> List[Dict]:
        """提取引用信息"""
        citations = []

        for i, result in enumerate(search_results, 1):
            # 检查答案中是否引用了该来源
            if f"来源{i}" in answer or result['source_id'] in answer:
                citations.append({
                    'source_id': result['source_id'],
                    'section': result['section'],
                    'relevance_score': result['score']
                })

        return citations

    def _generate_suggestions(
            self,
            question: str,
            answer: str,
            search_results: List[Dict]
    ) -> List[str]:
        """生成后续建议问题"""
        suggestions = []

        # 基于问题类型生成建议
        if "是什么" in question or "什么是" in question:
            suggestions.append(f"能详细解释一下{question.replace('是什么', '').replace('什么是', '')}的应用场景吗？")

        if "如何" in question or "怎么" in question:
            suggestions.append("有哪些具体的实例或案例吗？")

        if any(name in question for name in ["李睿凡", "研究者", "作者"]):
            suggestions.append("这位研究者的主要合作者有哪些？")
            suggestions.append("他们的研究有什么创新点？")

        # 基于检索结果生成建议
        if search_results:
            # 提取高频关键词
            keywords = self._extract_keywords_from_results(search_results)
            if keywords:
                suggestions.append(f"能介绍一下{keywords[0]}的更多细节吗？")

        return suggestions[:3]  # 最多返回3个建议

    def _extract_keywords_from_results(self, results: List[Dict]) -> List[str]:
        """从检索结果中提取关键词"""
        # 简单的关键词提取（实际可以用更复杂的方法）
        text = " ".join(r['text'] for r in results[:3])

        # 这里可以集成jieba等中文分词工具
        # 现在只是简单示例
        important_words = ['BERT', 'Transformer', '神经网络', '深度学习', '计算机视觉']

        found_keywords = [w for w in important_words if w in text]
        return found_keywords

    def _generate_fallback_answer(self, context: QAContext) -> str:
        """生成降级答案"""
        if not context.search_results:
            return "抱歉，我没有找到与您问题相关的信息。请尝试换个方式提问，或者提供更多上下文。"
        else:
            return "根据检索到的信息，我无法准确回答您的问题。找到的内容可能相关性不够高。建议您：\n1. 尝试使用更具体的关键词\n2. 将问题分解为更小的部分\n3. 查看原始文档获取更多信息"

    def _update_conversation_history(
            self,
            conversation_id: str,
            question: str,
            answer: str
    ):
        """更新对话历史"""
        self.conversation_memory[conversation_id].append({
            'question': question,
            'answer': answer,
            'timestamp': time.time()
        })

        # 保持历史记录在合理范围内
        if len(self.conversation_memory[conversation_id]) > 20:
            self.conversation_memory[conversation_id] = \
                self.conversation_memory[conversation_id][-20:]

    def get_conversation_history(self, conversation_id: str) -> List[Dict]:
        """获取对话历史"""
        return self.conversation_memory.get(conversation_id, [])

    def clear_conversation(self, conversation_id: str):
        """清空对话历史"""
        if conversation_id in self.conversation_memory:
            del self.conversation_memory[conversation_id]


class SimpleQAInterface:
    """简单的问答接口"""

    def __init__(self):
        self.qa_system = AcademicQASystem()
        self.current_conversation = "default"

    def ask(self, question: str, **kwargs) -> Dict[str, Any]:
        """简单的问答接口"""
        response = self.qa_system.answer_question(
            question=question,
            conversation_id=self.current_conversation,
            **kwargs
        )

        return {
            'answer': response.answer,
            'citations': response.citations,
            'confidence': f"{response.confidence:.2%}",
            'search_quality': response.search_quality,
            'suggestions': response.suggestions
        }

    def start_new_conversation(self):
        """开始新对话"""
        import uuid
        self.current_conversation = str(uuid.uuid4())
        return self.current_conversation

    def get_history(self) -> List[Dict]:
        """获取当前对话历史"""
        return self.qa_system.get_conversation_history(self.current_conversation)


def test_qa_system():
    """测试问答系统"""
    print("初始化问答系统...")
    qa_interface = SimpleQAInterface()

    # 测试问题集
    test_questions = [
        {
            'question': "李睿凡的主要研究方向是什么？",
            'user_level': 'student'
        },
        {
            'question': "BERT模型在NLP中有哪些应用？",
            'user_level': 'researcher'
        },
        {
            'question': "团队最近的创新成果有哪些？",
            'user_level': 'general'
        },
        {
            'question': "图像描述生成的主要技术挑战是什么？",
            'user_level': 'student'
        }
    ]

    print("\n开始测试问答...")
    print("=" * 80)

    for i, test_case in enumerate(test_questions, 1):
        print(f"\n问题 {i}: {test_case['question']}")
        print(f"用户级别: {test_case['user_level']}")
        print("-" * 40)

        try:
            result = qa_interface.ask(
                question=test_case['question'],
                user_level=test_case['user_level'],
                top_k=3
            )

            print(f"回答: {result['answer']}")
            print(f"\n置信度: {result['confidence']}")
            print(f"检索质量: {result['search_quality']}")

            if result['citations']:
                print("\n引用来源:")
                for cite in result['citations']:
                    print(f"  - {cite['source_id']} ({cite['section']})")

            if result['suggestions']:
                print("\n推荐问题:")
                for j, suggestion in enumerate(result['suggestions'], 1):
                    print(f"  {j}. {suggestion}")

        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()

        print("=" * 80)

    # 测试多轮对话
    print("\n\n测试多轮对话...")
    print("=" * 80)

    qa_interface.start_new_conversation()

    multi_turn_questions = [
        "什么是图像描述生成？",
        "它有哪些主要的技术方法？",
        "这些方法的优缺点是什么？"
    ]

    for question in multi_turn_questions:
        print(f"\n用户: {question}")
        result = qa_interface.ask(question, user_level='student')
        print(f"助手: {result['answer'][:200]}...")
        time.sleep(1)  # 避免API调用过快

    # 显示对话历史
    print("\n\n对话历史:")
    history = qa_interface.get_history()
    for turn in history:
        print(f"Q: {turn['question']}")
        print(f"A: {turn['answer'][:100]}...\n")


if __name__ == "__main__":
    # 注意：运行前需要设置正确的API Key
    print("请确保已设置正确的LLM_API_KEY")
    test_qa_system()