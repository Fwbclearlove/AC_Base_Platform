# src/05_retrieval_system.py
import os
os.environ["TRANSFORMERS_TRUST_REMOTE_CODE"] = "1"
os.environ["TRANSFORMERS_USE_SAFETENSORS"] = "1"
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
import json
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
import logging
from collections import defaultdict
import time
from sentence_transformers import SentenceTransformer

from config import (
    PROCESSED_DATA_DIR,
    VECTOR_INDICES_DIR,
    METADATA_STORE_PATH,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    VERBOSE
)

# 设置日志
logging.basicConfig(level=logging.INFO if VERBOSE else logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class SearchQuery:
    """搜索查询对象"""
    query_text: str
    search_type: str = "hybrid"  # hybrid, semantic, keyword, academic_profile
    top_k: int = 10
    filters: Optional[Dict] = None
    boost_recent: bool = False
    expand_context: bool = True


@dataclass
class SearchResult:
    """搜索结果对象"""
    chunk_id: str
    source_id: str
    text: str
    score: float
    section_type: str
    section_title: str
    metadata: Dict
    context_before: Optional[str] = None
    context_after: Optional[str] = None
    highlights: Optional[List[str]] = None


class MultiLayerRetriever:
    """多层次检索系统"""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None
        self.indices = {}
        self.metadata_store = {}
        self.summaries = {}
        self.author_profiles = {}
        self.domain_analysis = {}

        # 检索策略权重
        self.layer_weights = {
            'summary': 0.3,  # 概要层权重
            'main': 0.5,  # 主索引权重
            'section': 0.2  # 章节索引权重
        }

    def load_resources(self):
        """加载所有必要资源"""
        logger.info("加载检索资源...")

        # 1. 加载向量模型
        self._load_embedding_model()

        # 2. 加载FAISS索引
        self._load_indices()

        # 3. 加载元数据
        self._load_metadata()

        # 4. 加载知识库数据
        self._load_knowledge_base()

        logger.info("检索资源加载完成")

    def _load_embedding_model(self):
        """加载embedding模型"""
        logger.info(f"加载模型: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL, device=self.device)

    def _load_indices(self):
        """加载所有FAISS索引"""
        index_files = list(VECTOR_INDICES_DIR.glob("*.faiss"))

        for index_file in index_files:
            index_name = index_file.stem.replace("_index", "")
            logger.info(f"加载索引: {index_name}")
            self.indices[index_name] = faiss.read_index(str(index_file))

        logger.info(f"成功加载 {len(self.indices)} 个索引")

    def _load_metadata(self):
        """加载元数据"""
        with open(METADATA_STORE_PATH, 'r', encoding='utf-8') as f:
            self.metadata_store = json.load(f)
        logger.info(f"加载元数据完成")

    def _load_knowledge_base(self):
        """加载知识库相关数据"""
        # 加载概要数据
        summaries_path = PROCESSED_DATA_DIR / "document_summaries.json"
        if summaries_path.exists():
            with open(summaries_path, 'r', encoding='utf-8') as f:
                self.summaries = json.load(f)

        # 加载学术画像
        profiles_path = PROCESSED_DATA_DIR / "direct_academic_profiles" / "author_profiles.json"
        if profiles_path.exists():
            with open(profiles_path, 'r', encoding='utf-8') as f:
                self.author_profiles = json.load(f)

        # 加载领域分析
        domain_path = PROCESSED_DATA_DIR / "direct_academic_profiles" / "domain_analysis.json"
        if domain_path.exists():
            with open(domain_path, 'r', encoding='utf-8') as f:
                self.domain_analysis = json.load(f)

    def search(self, query: SearchQuery) -> List[SearchResult]:
        """执行搜索"""
        logger.info(f"执行搜索: {query.query_text[:50]}... (类型: {query.search_type})")

        # 根据查询类型选择策略
        if query.search_type == "academic_profile":
            return self._search_academic_profile(query)
        elif query.search_type == "semantic":
            return self._search_semantic(query)
        elif query.search_type == "keyword":
            return self._search_keyword(query)
        else:  # hybrid
            return self._search_hybrid(query)

    def _search_hybrid(self, query: SearchQuery) -> List[SearchResult]:
        """混合检索策略 - 修复版"""
        # 1. 向量化查询
        query_embedding = self._encode_query(query.query_text)

        # 2. 多层检索
        all_results = []

        # 概要层检索
        if 'summary' in self.indices:
            summary_results = self._search_index(
                query_embedding,
                'summary',
                top_k=query.top_k * 2
            )
            for idx, score in summary_results:
                result = self._create_search_result(idx, score * self.layer_weights['summary'], 'summary')
                if result:
                    all_results.append(result)

        # 主索引检索
        if 'main' in self.indices:
            main_results = self._search_index(
                query_embedding,
                'main',
                top_k=query.top_k * 3
            )
            for idx, score in main_results:
                result = self._create_search_result(idx, score * self.layer_weights['main'], 'main')
                if result:
                    all_results.append(result)

        # 章节索引检索（根据查询内容选择）
        section_type = self._infer_section_type(query.query_text)
        if section_type and f'section_{section_type}' in self.indices:
            section_results = self._search_index(
                query_embedding,
                f'section_{section_type}',
                top_k=query.top_k
            )
            for idx, score in section_results:
                result = self._create_search_result(
                    idx,
                    score * self.layer_weights['section'],
                    f'section_{section_type}'
                )
                if result:
                    all_results.append(result)

        # 3. 去重和排序
        seen_chunks = set()
        unique_results = []
        for result in all_results:
            if result.chunk_id not in seen_chunks:
                seen_chunks.add(result.chunk_id)
                unique_results.append(result)

        # 4. 排序
        unique_results.sort(key=lambda x: x.score, reverse=True)

        # 5. 重排序
        reranked_results = self._rerank_results(unique_results, query)

        # 6. 扩展上下文
        if query.expand_context:
            reranked_results = self._expand_context(reranked_results)

        # 7. 添加高亮
        reranked_results = self._add_highlights(reranked_results, query.query_text)

        return reranked_results[:query.top_k]

    def _search_semantic(self, query: SearchQuery) -> List[SearchResult]:
        """纯语义检索"""
        query_embedding = self._encode_query(query.query_text)

        # 主要使用main索引
        if 'main' not in self.indices:
            logger.warning("主索引不存在")
            return []

        results = self._search_index(query_embedding, 'main', top_k=query.top_k * 2)

        # 转换为SearchResult对象
        search_results = []
        for idx, score in results:
            result = self._create_search_result(idx, score, 'main')
            if result:
                search_results.append(result)

        return search_results[:query.top_k]

    def _search_academic_profile(self, query: SearchQuery) -> List[SearchResult]:
        """学术画像专门检索"""
        query_lower = query.query_text.lower()
        results = []

        # 1. 识别查询意图
        if any(word in query_lower for word in ['研究方向', '研究领域', 'research area', 'domain']):
            results = self._search_research_domains(query)
        elif any(word in query_lower for word in ['合作', 'collaboration', '团队', 'team']):
            results = self._search_collaboration_network(query)
        elif any(word in query_lower for word in ['创新', 'innovation', '贡献', 'contribution']):
            results = self._search_innovations(query)
        elif any(word in query_lower for word in ['方法', 'method', '技术', 'technique']):
            results = self._search_technical_methods(query)
        else:
            # 通用作者搜索
            results = self._search_author_general(query)

        return results

    def _search_research_domains(self, query: SearchQuery) -> List[SearchResult]:
        """搜索研究领域信息"""
        results = []

        # 从查询中提取人名或领域名
        entities = self._extract_entities(query.query_text)

        # 如果识别到人名，返回该人的研究领域
        for author_name in entities.get('authors', []):
            if author_name in self.author_profiles:
                profile = self.author_profiles[author_name]
                domains = profile['research_focus']['primary_domains']

                # 创建一个合成的搜索结果
                result_text = f"{author_name}的主要研究领域包括：\n"
                for i, domain_info in enumerate(domains[:5], 1):
                    result_text += f"{i}. {domain_info['domain']} (论文数: {domain_info['count']})\n"

                result = SearchResult(
                    chunk_id=f"profile_{author_name}_domains",
                    source_id=f"author_profile_{author_name}",
                    text=result_text,
                    score=1.0,
                    section_type="author_profile",
                    section_title="研究领域分析",
                    metadata={
                        'author': author_name,
                        'profile_type': 'research_domains',
                        'data_source': 'author_profiles'
                    }
                )
                results.append(result)

        # 如果识别到领域名，返回该领域的信息
        for domain in entities.get('domains', []):
            if domain in self.domain_analysis:
                domain_info = self.domain_analysis[domain]

                result_text = f"领域「{domain}」的研究概况：\n"
                result_text += f"- 论文总数: {domain_info['paper_count']}\n"
                result_text += f"- 活跃研究者: {domain_info['author_count']}人\n"
                result_text += f"- 活跃度: {domain_info['activity_level']}\n"

                if domain_info['top_methods']:
                    result_text += "\n主要研究方法:\n"
                    for method in domain_info['top_methods'][:3]:
                        result_text += f"- {method['method']} (使用{method['count']}次)\n"

                result = SearchResult(
                    chunk_id=f"domain_{domain}",
                    source_id=f"domain_analysis",
                    text=result_text,
                    score=0.95,
                    section_type="domain_analysis",
                    section_title="领域分析",
                    metadata={
                        'domain': domain,
                        'profile_type': 'domain_overview',
                        'data_source': 'domain_analysis'
                    }
                )
                results.append(result)

        # 如果没有特定实体，进行语义搜索
        if not results:
            results = self._search_semantic(query)

        return results

    def _search_collaboration_network(self, query: SearchQuery) -> List[SearchResult]:
        """搜索合作网络信息"""
        results = []
        entities = self._extract_entities(query.query_text)

        for author_name in entities.get('authors', []):
            if author_name in self.author_profiles:
                profile = self.author_profiles[author_name]
                collab_info = profile['collaboration_network']

                result_text = f"{author_name}的合作网络分析：\n"
                result_text += f"- 合作者数量: {collab_info['collaboration_strength']}人\n"
                result_text += f"- 网络级别: {collab_info['network_level']}\n"

                if collab_info['direct_collaborators']:
                    result_text += f"\n主要合作者:\n"
                    for collaborator in collab_info['direct_collaborators'][:8]:
                        result_text += f"- {collaborator}\n"

                if collab_info['institutional_networks']:
                    result_text += f"\n合作机构:\n"
                    for inst in collab_info['institutional_networks'][:5]:
                        result_text += f"- {inst}\n"

                result = SearchResult(
                    chunk_id=f"profile_{author_name}_collaboration",
                    source_id=f"author_profile_{author_name}",
                    text=result_text,
                    score=0.98,
                    section_type="author_profile",
                    section_title="合作网络",
                    metadata={
                        'author': author_name,
                        'profile_type': 'collaboration_network',
                        'data_source': 'author_profiles'
                    }
                )
                results.append(result)

        return results

    def _search_innovations(self, query: SearchQuery) -> List[SearchResult]:
        """搜索创新贡献信息"""
        results = []
        entities = self._extract_entities(query.query_text)

        for author_name in entities.get('authors', []):
            if author_name in self.author_profiles:
                profile = self.author_profiles[author_name]
                innovation_info = profile['innovation_profile']

                result_text = f"{author_name}的创新贡献分析：\n"
                result_text += f"- 创新水平: {innovation_info['innovation_level']}\n"
                result_text += f"- 贡献总数: {innovation_info['total_contributions']}\n"
                result_text += f"- 技术新颖性: {innovation_info['total_novelty_claims']}项\n"
                result_text += f"- 创新强度: {innovation_info['innovation_strength']}\n"

                if innovation_info['datasets_expertise']:
                    result_text += f"\n数据集专长:\n"
                    for dataset in innovation_info['datasets_expertise'][:5]:
                        result_text += f"- {dataset}\n"

                result = SearchResult(
                    chunk_id=f"profile_{author_name}_innovations",
                    source_id=f"author_profile_{author_name}",
                    text=result_text,
                    score=0.97,
                    section_type="author_profile",
                    section_title="创新贡献",
                    metadata={
                        'author': author_name,
                        'profile_type': 'innovation_analysis',
                        'data_source': 'author_profiles'
                    }
                )
                results.append(result)

        return results

    def _search_technical_methods(self, query: SearchQuery) -> List[SearchResult]:
        """搜索技术方法信息"""
        results = []

        # 先尝试语义搜索method相关内容
        query_embedding = self._encode_query(query.query_text)

        # 优先搜索method章节
        if 'section_method' in self.indices:
            method_results = self._search_index(
                query_embedding,
                'section_method',
                top_k=query.top_k * 2
            )

            for idx, score in method_results:
                result = self._create_search_result(idx, score, 'section_method')
                if result:
                    results.append(result)

        # 补充experiment章节
        if 'section_experiment' in self.indices and len(results) < query.top_k:
            exp_results = self._search_index(
                query_embedding,
                'section_experiment',
                top_k=query.top_k
            )

            for idx, score in exp_results:
                result = self._create_search_result(idx, score, 'section_experiment')
                if result:
                    results.append(result)

        return results[:query.top_k]

    def _search_author_general(self, query: SearchQuery) -> List[SearchResult]:
        """通用作者信息搜索"""
        results = []
        entities = self._extract_entities(query.query_text)

        for author_name in entities.get('authors', []):
            if author_name in self.author_profiles:
                profile = self.author_profiles[author_name]

                # 生成综合画像
                result_text = f"【{author_name}的学术画像】\n\n"

                # 基本信息
                basic = profile['basic_info']
                result_text += f"基本信息:\n"
                result_text += f"- 标准化姓名: {basic['standardized_name']}\n"
                result_text += f"- 论文总数: {basic['total_papers']}\n"
                result_text += f"- 产出水平: {basic['productivity_level']}\n"
                result_text += f"- 合作人数: {basic['collaboration_count']}\n"

                # 研究重点
                result_text += f"\n研究重点:\n"
                domains = profile['research_focus']['primary_domains']
                for domain_info in domains[:3]:
                    result_text += f"- {domain_info['domain']} ({domain_info['count']}篇)\n"

                # 创新水平
                innovation = profile['innovation_profile']
                result_text += f"\n创新概况:\n"
                result_text += f"- 创新级别: {innovation['innovation_level']}\n"
                result_text += f"- 贡献数量: {innovation['total_contributions']}\n"

                # 合作网络
                collab = profile['collaboration_network']
                result_text += f"\n合作网络:\n"
                result_text += f"- 网络级别: {collab['network_level']}\n"
                result_text += f"- 合作者数: {collab['collaboration_strength']}\n"

                result = SearchResult(
                    chunk_id=f"profile_{author_name}_general",
                    source_id=f"author_profile_{author_name}",
                    text=result_text,
                    score=1.0,
                    section_type="author_profile",
                    section_title="综合画像",
                    metadata={
                        'author': author_name,
                        'profile_type': 'general_profile',
                        'data_source': 'author_profiles'
                    }
                )
                results.append(result)

        return results

    def _search_keyword(self, query: SearchQuery) -> List[SearchResult]:
        """关键词检索（基于文本匹配）"""
        keywords = query.query_text.lower().split()
        results = []

        # 在所有文本块中搜索
        if 'main' in self.metadata_store:
            texts = self.metadata_store['main']['texts']
            metadata_list = self.metadata_store['main']['metadata']

            for i, text in enumerate(texts):
                text_lower = text.lower()
                # 计算关键词匹配分数
                match_score = sum(1 for kw in keywords if kw in text_lower) / len(keywords)

                if match_score > 0:
                    result = SearchResult(
                        chunk_id=metadata_list[i]['chunk_id'],
                        source_id=metadata_list[i]['source_id'],
                        text=text,
                        score=match_score,
                        section_type=metadata_list[i]['section_type'],
                        section_title=metadata_list[i]['section_title'],
                        metadata=metadata_list[i]
                    )
                    results.append(result)

        # 按匹配分数排序
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:query.top_k]

    def _encode_query(self, query_text: str) -> np.ndarray:
        """编码查询文本"""
        embedding = self.model.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embedding[0]

    def _search_index(self, query_embedding: np.ndarray, index_name: str,
                      top_k: int) -> List[Tuple[int, float]]:
        """在指定索引中搜索"""
        if index_name not in self.indices:
            logger.warning(f"索引 {index_name} 不存在")
            return []

        index = self.indices[index_name]

        # 搜索
        scores, indices = index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            top_k
        )

        # 返回(index, score)对
        results = []
        for i in range(len(indices[0])):
            if indices[0][i] >= 0:  # 有效索引
                results.append((indices[0][i], scores[0][i]))

        return results

    def _create_search_result(self, idx: int, score: float,
                              index_name: str) -> Optional[SearchResult]:
        """创建搜索结果对象 - 修复版，处理不同索引的数据结构"""
        if index_name not in self.metadata_store:
            logger.warning(f"元数据中找不到索引: {index_name}")
            return None

        metadata_info = self.metadata_store[index_name]

        # 检查数据结构并适配
        if index_name == 'summary':
            # 概要层有不同的数据结构
            if 'doc_ids' not in metadata_info or 'summaries' not in metadata_info:
                logger.warning(f"概要层元数据结构不正确")
                return None

            if idx >= len(metadata_info['doc_ids']):
                return None

            doc_id = metadata_info['doc_ids'][idx]
            summary = metadata_info['summaries'].get(doc_id, {})

            # 构建概要文本
            text = f"标题: {summary.get('title', '未知')}\n"
            text += f"类型: {summary.get('document_type', '未知')}\n"
            text += f"摘要: {summary.get('summary', '无摘要')}\n"

            # 添加主要内容
            if summary.get('main_topic'):
                text += f"主要研究领域: {summary.get('main_topic')}\n"
            if summary.get('methodology'):
                text += f"方法: {summary.get('methodology')}\n"
            if summary.get('key_innovations'):
                innovations = summary.get('key_innovations', [])
                if innovations and isinstance(innovations, list):
                    text += f"主要创新: {', '.join(innovations[:3])}\n"

            return SearchResult(
                chunk_id=f"summary_{doc_id}",
                source_id=doc_id,
                text=text,
                score=float(score),
                section_type="summary",
                section_title="文档概要",
                metadata={
                    'doc_id': doc_id,
                    'document_type': summary.get('document_type', 'unknown'),
                    'index_type': 'summary'
                }
            )
        else:
            # 其他索引使用标准结构
            if 'metadata' not in metadata_info or 'texts' not in metadata_info:
                logger.warning(f"索引 {index_name} 的元数据结构不正确")
                return None

            if idx >= len(metadata_info['metadata']):
                return None

            metadata = metadata_info['metadata'][idx]
            text = metadata_info['texts'][idx]

            return SearchResult(
                chunk_id=metadata.get('chunk_id', f"{index_name}_{idx}"),
                source_id=metadata.get('source_id', 'unknown'),
                text=text,
                score=float(score),
                section_type=metadata.get('section_type', 'unknown'),
                section_title=metadata.get('section_title', ''),
                metadata=metadata
            )

    def _get_layer_weight(self, layer_name: str) -> float:
        """获取层权重"""
        if layer_name == 'summary':
            return self.layer_weights['summary']
        elif layer_name == 'main':
            return self.layer_weights['main']
        elif layer_name.startswith('section_'):
            return self.layer_weights['section']
        else:
            return 0.5

    def _infer_section_type(self, query_text: str) -> Optional[str]:
        """推断查询对应的章节类型"""
        query_lower = query_text.lower()

        section_keywords = {
            'method': ['方法', 'method', 'approach', '算法', 'algorithm', '模型', 'model'],
            'experiment': ['实验', 'experiment', '评估', 'evaluation', '测试', 'test', '结果', 'result'],
            'introduction': ['介绍', 'introduction', '背景', 'background', '动机', 'motivation'],
            'conclusion': ['结论', 'conclusion', '总结', 'summary', '展望', 'future'],
            'related_work': ['相关工作', 'related work', '研究现状', 'literature'],
        }

        for section_type, keywords in section_keywords.items():
            if any(kw in query_lower for kw in keywords):
                return section_type

        return None

    def _rerank_results(self, results: List[SearchResult],
                        query: SearchQuery) -> List[SearchResult]:
        """重排序结果"""
        # 这里可以集成更复杂的重排序逻辑
        # 例如使用BGE-reranker或其他模型

        # 简单的启发式重排序
        for result in results:
            # 提升包含查询关键词的结果
            query_words = set(query.query_text.lower().split())
            result_words = set(result.text.lower().split())
            keyword_overlap = len(query_words & result_words) / len(query_words)
            result.score *= (1 + keyword_overlap * 0.2)

            # 根据章节类型调整分数
            if result.section_type == 'abstract':
                result.score *= 1.1
            elif result.section_type == 'conclusion':
                result.score *= 1.05

            # 如果启用了boost_recent
            if query.boost_recent and 'year' in result.metadata:
                year = result.metadata.get('year', 2020)
                recency_boost = (year - 2018) / 10  # 越新的文档分数越高
                result.score *= (1 + max(0, recency_boost))

        # 重新排序
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def _expand_context(self, results: List[SearchResult]) -> List[SearchResult]:
        """扩展搜索结果的上下文"""
        # 获取相邻块的内容
        for result in results:
            source_id = result.source_id
            chunk_index = result.metadata.get('chunk_index', 0)

            # 查找前后文
            if 'main' in self.metadata_store:
                all_metadata = self.metadata_store['main']['metadata']
                all_texts = self.metadata_store['main']['texts']

                # 找到同一文档的所有块
                doc_chunks = [
                    (i, m) for i, m in enumerate(all_metadata)
                    if m['source_id'] == source_id
                ]
                doc_chunks.sort(key=lambda x: x[1]['chunk_index'])

                # 找到当前块的位置
                current_pos = None
                for pos, (idx, meta) in enumerate(doc_chunks):
                    if meta['chunk_index'] == chunk_index:
                        current_pos = pos
                        break

                if current_pos is not None:
                    # 获取前文
                    if current_pos > 0:
                        prev_idx = doc_chunks[current_pos - 1][0]
                        result.context_before = all_texts[prev_idx][:200] + "..."

                    # 获取后文
                    if current_pos < len(doc_chunks) - 1:
                        next_idx = doc_chunks[current_pos + 1][0]
                        result.context_after = "..." + all_texts[next_idx][:200]

        return results

    def _add_highlights(self, results: List[SearchResult],
                        query_text: str) -> List[SearchResult]:
        """添加高亮显示"""
        query_words = set(query_text.lower().split())

        for result in results:
            highlights = []
            text_lower = result.text.lower()

            # 找到包含查询词的句子
            sentences = result.text.split('。')
            for sentence in sentences:
                if any(word in sentence.lower() for word in query_words):
                    highlights.append(sentence.strip())

            result.highlights = highlights[:3]  # 最多3个高亮句子

        return results

    def _extract_entities(self, query_text: str) -> Dict[str, List[str]]:
        """从查询中提取实体（人名、领域等）- 改进版"""
        entities = {
            'authors': [],
            'domains': [],
            'methods': []
        }

        # 标准化查询文本
        normalized_query = query_text

        # 处理"李睿凡"的各种写法
        ruifan_variations = [
            "李睿凡", "Ruifan Li", "Li Ruifan", "ruifan li", "li ruifan",
            "李睿", "睿凡"
        ]

        for variation in ruifan_variations:
            if variation.lower() in query_text.lower():
                entities['authors'].append("李睿凡")  # 使用标准化形式
                break

        # 检查其他已知作者
        if not entities['authors']:
            for author_name in self.author_profiles.keys():
                if author_name in query_text or author_name.lower() in query_text.lower():
                    entities['authors'].append(author_name)

        # 识别可能的领域名
        for domain_name in self.domain_analysis.keys():
            if domain_name in query_text or domain_name.lower() in query_text.lower():
                entities['domains'].append(domain_name)

        # 如果没有识别到，尝试一些启发式规则
        if not entities['authors']:
            # 查找"XXX的"模式
            import re
            pattern = r'(\S{2,4})的'
            matches = re.findall(pattern, query_text)
            for match in matches:
                # 检查是否可能是人名
                if match in self.author_profiles:
                    entities['authors'].append(match)

        return entities

    def get_statistics(self) -> Dict[str, Any]:
        """获取检索系统统计信息"""
        stats = {
            'loaded_indices': list(self.indices.keys()),
            'index_sizes': {
                name: index.ntotal for name, index in self.indices.items()
            },
            'total_documents': len(self.summaries),
            'total_authors': len(self.author_profiles),
            'total_domains': len(self.domain_analysis),
            'metadata_layers': list(self.metadata_store.keys())
        }
        return stats

    def debug_metadata_structure(self):
        """调试：打印元数据结构"""
        print("\n=== 元数据结构调试 ===")
        for index_name, metadata in self.metadata_store.items():
            print(f"\n索引: {index_name}")
            print(f"键值: {list(metadata.keys())}")
            if 'metadata' in metadata:
                print(f"  - metadata长度: {len(metadata['metadata'])}")
            if 'texts' in metadata:
                print(f"  - texts长度: {len(metadata['texts'])}")
            if 'doc_ids' in metadata:
                print(f"  - doc_ids长度: {len(metadata['doc_ids'])}")
            if 'index_type' in metadata:
                print(f"  - index_type: {metadata['index_type']}")


class SearchEngine:
    """搜索引擎的便捷封装"""

    def __init__(self, device: str = "cpu"):
        self.retriever = MultiLayerRetriever(device=device)
        self.is_loaded = False

    def initialize(self):
        """初始化搜索引擎"""
        if not self.is_loaded:
            self.retriever.load_resources()
            self.is_loaded = True
            logger.info("搜索引擎初始化完成")

    def search(self, query_text: str, search_type: str = "hybrid",
               top_k: int = 10, **kwargs) -> List[Dict]:
        """执行搜索并返回格式化结果"""
        self.initialize()

        # 创建查询对象
        query = SearchQuery(
            query_text=query_text,
            search_type=search_type,
            top_k=top_k,
            **kwargs
        )

        # 执行搜索
        results = self.retriever.search(query)

        # 格式化输出
        formatted_results = []
        for result in results:
            formatted_results.append({
                'chunk_id': result.chunk_id,
                'source_id': result.source_id,
                'text': result.text,
                'score': result.score,
                'section': f"{result.section_title} ({result.section_type})",
                'highlights': result.highlights,
                'context': {
                    'before': result.context_before,
                    'after': result.context_after
                },
                'metadata': result.metadata
            })

        return formatted_results

    def search_author(self, author_name: str) -> List[Dict]:
        """搜索作者信息"""
        return self.search(
            f"{author_name}的学术画像",
            search_type="academic_profile"
        )

    def search_domain(self, domain: str) -> List[Dict]:
        """搜索领域信息"""
        return self.search(
            f"{domain}领域的研究",
            search_type="academic_profile"
        )

    def search_method(self, method: str) -> List[Dict]:
        """搜索方法信息"""
        return self.search(
            f"{method}方法",
            search_type="hybrid"
        )


def test_retrieval_system():
    """测试检索系统 - 增强版"""
    print("初始化检索系统...")
    engine = SearchEngine(device="cpu")
    engine.initialize()

    # 调试元数据结构
    print("\n调试元数据结构...")
    engine.retriever.debug_metadata_structure()

    # 打印统计信息
    stats = engine.retriever.get_statistics()
    print(f"\n系统统计:")
    print(f"- 加载索引: {len(stats['loaded_indices'])}个")
    print(f"- 索引列表: {', '.join(stats['loaded_indices'])}")
    print(f"- 总向量数: {sum(stats['index_sizes'].values())}")
    print(f"- 文档数: {stats['total_documents']}")
    print(f"- 作者数: {stats['total_authors']}")

    # 先测试简单查询
    print("\n测试基础功能...")

    # 1. 测试语义搜索
    print("\n1. 测试语义搜索:")
    try:
        results = engine.search("机器学习", search_type="semantic", top_k=2)
        print(f"找到 {len(results)} 个结果")
        if results:
            print(f"第一个结果得分: {results[0]['score']:.3f}")
    except Exception as e:
        print(f"语义搜索出错: {e}")

    # 2. 测试学术画像（如果有数据）
    if stats['total_authors'] > 0:
        print("\n2. 测试学术画像搜索:")
        # 获取第一个作者名字用于测试
        first_author = list(engine.retriever.author_profiles.keys())[0]
        print(f"搜索作者: {first_author}")
        try:
            results = engine.search_author(first_author)
            if results:
                print(f"找到 {len(results)} 个结果")
                print(f"第一个结果: {results[0]['text'][:100]}...")
        except Exception as e:
            print(f"学术画像搜索出错: {e}")

    # 3. 测试完整查询集
    test_queries = [
        ("李睿凡的研究方向", "academic_profile"),
        ("BERT在NLP中的应用", "hybrid"),
        ("实验使用了哪些数据集", "hybrid"),
        ("计算机视觉", "semantic"),
    ]

    print("\n测试查询集:")
    for query, search_type in test_queries:
        try:
            print(f"\n查询: {query} (类型: {search_type})")
            results = engine.search(query, search_type=search_type, top_k=2)

            if results:
                for i, result in enumerate(results, 1):
                    print(f"\n结果 {i}:")
                    print(f"- 来源: {result['source_id']}")
                    print(f"- 章节: {result['section']}")
                    print(f"- 得分: {result['score']:.3f}")
                    print(f"- 内容预览: {result['text'][:150]}...")
            else:
                print("未找到相关结果")

        except Exception as e:
            print(f"查询出错: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    test_retrieval_system()