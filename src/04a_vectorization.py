# src/04a_vectorization_enhanced.py
import os
os.environ["TRANSFORMERS_TRUST_REMOTE_CODE"] = "1"
os.environ["TRANSFORMERS_USE_SAFETENSORS"] = "1"
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass, asdict

from config import (
    PROCESSED_DATA_DIR,
    SMART_CHUNKS_PATH,  # 使用新的智能分块
    VECTOR_INDICES_DIR,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    EMBEDDING_BATCH_SIZE,
    METADATA_STORE_PATH,
    VERBOSE
)

# 设置日志
logging.basicConfig(level=logging.INFO if VERBOSE else logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """块元数据结构"""
    chunk_id: str
    source_id: str
    section_type: str
    section_title: str
    chunk_index: int
    chunk_length: int
    importance_weight: float
    text_source: str  # cleaned/extracted
    document_type: str  # patent/academic_paper/unknown

    def to_dict(self):
        return asdict(self)


class EnhancedVectorizer:
    """增强的向量化系统"""

    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str = "cpu"):
        """
        初始化向量化器

        Args:
            model_name: Embedding模型名称
            device: 运行设备
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.dimension = EMBEDDING_DIMENSION

        # 索引存储
        self.indices = {}
        self.metadata_store = {}

    def load_model(self):
        """加载Embedding模型"""
        logger.info(f"加载模型: {self.model_name}")
        logger.info(f"使用设备: {self.device}")

        self.model = SentenceTransformer(self.model_name, device=self.device)

        # 验证维度
        test_embedding = self.model.encode(["test"], show_progress_bar=False)
        actual_dim = test_embedding.shape[1]

        if actual_dim != self.dimension:
            logger.warning(f"实际维度 {actual_dim} 与配置维度 {self.dimension} 不匹配，使用实际维度")
            self.dimension = actual_dim

        logger.info(f"模型加载完成，向量维度: {self.dimension}")

    def load_chunks_data(self) -> Dict[str, Any]:
        """加载智能分块数据"""
        logger.info("加载分块数据...")

        # 优先使用智能分块
        chunks_path = SMART_CHUNKS_PATH if SMART_CHUNKS_PATH.exists() else PROCESSED_DATA_DIR / "chunks.json"

        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)

        logger.info(f"加载了 {len(chunks_data)} 个文本块")
        return chunks_data

    def load_summaries_data(self) -> Optional[Dict[str, Any]]:
        """加载概要层数据（如果存在）"""
        summaries_path = PROCESSED_DATA_DIR / "document_summaries.json"

        if not summaries_path.exists():
            logger.warning("未找到概要层数据")
            return None

        logger.info("加载概要层数据...")
        with open(summaries_path, 'r', encoding='utf-8') as f:
            summaries = json.load(f)

        logger.info(f"加载了 {len(summaries)} 个文档概要")
        return summaries

    def prepare_chunk_texts_and_metadata(self, chunks_data: List[Dict]) -> tuple:
        """
        准备文本和元数据

        Returns:
            (texts, metadata_list)
        """
        texts = []
        metadata_list = []

        for chunk in chunks_data:
            # 提取文本
            text = chunk['text']

            # 构建增强文本（可选：添加章节信息增强检索）
            if chunk.get('section_title') and chunk['section_title'] != '全文':
                # 在文本前添加章节标题作为上下文
                enhanced_text = f"[{chunk['section_title']}] {text}"
            else:
                enhanced_text = text

            texts.append(enhanced_text)

            # 构建元数据
            metadata = ChunkMetadata(
                chunk_id=chunk['chunk_id'],
                source_id=chunk['source_id'],
                section_type=chunk.get('section_type', 'default'),
                section_title=chunk.get('section_title', ''),
                chunk_index=chunk.get('chunk_index', 0),
                chunk_length=chunk.get('chunk_length', len(text)),
                importance_weight=chunk.get('importance_weight', 0.7),
                text_source=chunk.get('document_metadata', {}).get('text_source', 'unknown'),
                document_type=chunk.get('document_metadata', {}).get('document_type', 'unknown')
            )

            metadata_list.append(metadata)

        return texts, metadata_list

    def convert_summary_to_text(self, summary: Dict) -> str:
        """
        将概要JSON转换为可向量化的文本

        Args:
            summary: 概要字典

        Returns:
            拼接后的文本
        """
        text_parts = []

        # 优先级高的字段
        priority_fields = [
            ('title', 2.0),
            ('summary', 1.5),
            ('main_topic', 1.3),
            ('key_innovations', 1.5),
            ('methodology', 1.2),
            ('keywords', 1.2),
            ('technical_concepts', 1.2),
            ('experimental_results', 1.0),
            ('conclusions', 1.1)
        ]

        for field, weight in priority_fields:
            if field in summary:
                value = summary[field]

                # 处理不同类型的值
                if isinstance(value, str) and value not in ["原文无此信息", "未在原文中明确提及", ""]:
                    text_parts.append(value)
                elif isinstance(value, list):
                    # 过滤无效值
                    valid_items = [
                        item for item in value
                        if item and item not in ["原文无此信息", "未在原文中明确提及"]
                    ]
                    if valid_items:
                        text_parts.append(" ".join(valid_items))

        return " ".join(text_parts)

    def create_faiss_index(self, embeddings: np.ndarray, index_type: str = "flat") -> faiss.Index:
        """
        创建FAISS索引

        Args:
            embeddings: 向量数组
            index_type: 索引类型 (flat/ivf/hnsw)

        Returns:
            FAISS索引对象
        """
        n_vectors = embeddings.shape[0]
        dimension = embeddings.shape[1]

        if index_type == "flat":
            # 精确搜索（适合小规模数据）
            index = faiss.IndexFlatIP(dimension)

        elif index_type == "ivf" and n_vectors > 1000:
            # IVF索引（适合中等规模）
            nlist = min(100, n_vectors // 10)  # 聚类数
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

            # 训练索引
            logger.info(f"训练IVF索引，聚类数: {nlist}")
            index.train(embeddings)

        elif index_type == "hnsw" and n_vectors > 5000:
            # HNSW索引（适合大规模，快速近似搜索）
            M = 32  # 连接数
            index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_INNER_PRODUCT)

        else:
            # 默认使用Flat
            index = faiss.IndexFlatIP(dimension)

        # 添加向量
        index.add(embeddings)

        logger.info(f"创建{index_type}索引完成，包含 {index.ntotal} 个向量")
        return index

    def vectorize_and_index(self):
        """执行完整的向量化和索引构建流程"""

        # 1. 加载模型
        if not self.model:
            self.load_model()

        # 2. 处理文本块
        chunks_data = self.load_chunks_data()

        if chunks_data:
            logger.info("=" * 60)
            logger.info("处理文本块...")

            # 准备数据
            texts, metadata_list = self.prepare_chunk_texts_and_metadata(chunks_data)

            # 批量编码
            logger.info(f"开始向量化 {len(texts)} 个文本块...")
            chunk_embeddings = self.model.encode(
                texts,
                batch_size=EMBEDDING_BATCH_SIZE,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True
            )

            # 创建主索引
            logger.info("创建主索引...")
            main_index = self.create_faiss_index(
                chunk_embeddings,
                index_type="ivf" if len(chunks_data) > 1000 else "flat"
            )

            # 保存索引和元数据
            self.indices['main'] = main_index
            self.metadata_store['main'] = {
                'metadata': [m.to_dict() for m in metadata_list],
                'texts': texts,  # 保存原始文本用于展示
                'index_type': 'chunk',
                'total_vectors': len(texts)
            }

            # 创建按章节类型的子索引（可选，用于特定检索）
            self._create_section_indices(chunk_embeddings, metadata_list, texts)

        # 3. 处理概要层（如果存在）
        summaries_data = self.load_summaries_data()

        if summaries_data:
            logger.info("=" * 60)
            logger.info("处理概要层...")

            summary_texts = []
            summary_ids = []

            for doc_id, summary in summaries_data.items():
                text = self.convert_summary_to_text(summary)
                if text:
                    summary_texts.append(text)
                    summary_ids.append(doc_id)

            if summary_texts:
                logger.info(f"向量化 {len(summary_texts)} 个文档概要...")
                summary_embeddings = self.model.encode(
                    summary_texts,
                    batch_size=EMBEDDING_BATCH_SIZE,
                    show_progress_bar=True,
                    normalize_embeddings=True,
                    convert_to_numpy=True
                )

                # 创建概要索引
                summary_index = self.create_faiss_index(summary_embeddings, index_type="flat")

                self.indices['summary'] = summary_index
                self.metadata_store['summary'] = {
                    'doc_ids': summary_ids,
                    'summaries': summaries_data,
                    'index_type': 'summary',
                    'total_vectors': len(summary_texts)
                }

        # 4. 保存所有索引和元数据
        self.save_indices()

        logger.info("=" * 60)
        logger.info("向量化完成！")
        self._print_statistics()

    def _create_section_indices(self, embeddings: np.ndarray,
                                metadata_list: List[ChunkMetadata],
                                texts: List[str]):
        """创建按章节类型分组的子索引"""

        section_groups = {}

        # 按章节类型分组
        for idx, metadata in enumerate(metadata_list):
            section_type = metadata.section_type
            if section_type not in section_groups:
                section_groups[section_type] = {
                    'indices': [],
                    'embeddings': [],
                    'metadata': [],
                    'texts': []
                }

            section_groups[section_type]['indices'].append(idx)
            section_groups[section_type]['embeddings'].append(embeddings[idx])
            section_groups[section_type]['metadata'].append(metadata)
            section_groups[section_type]['texts'].append(texts[idx])

        # 为每个章节类型创建索引（只为有足够数据的章节创建）
        for section_type, group_data in section_groups.items():
            if len(group_data['indices']) >= 10:  # 至少10个块才创建独立索引

                section_embeddings = np.array(group_data['embeddings'])
                section_index = self.create_faiss_index(section_embeddings, index_type="flat")

                index_name = f"section_{section_type}"
                self.indices[index_name] = section_index
                self.metadata_store[index_name] = {
                    'metadata': [m.to_dict() for m in group_data['metadata']],
                    'texts': group_data['texts'],
                    'index_type': 'section',
                    'section_type': section_type,
                    'total_vectors': len(group_data['indices'])
                }

                logger.info(f"创建章节索引: {section_type} ({len(group_data['indices'])} 个向量)")

    def save_indices(self):
        """保存所有索引和元数据"""
        logger.info("保存索引文件...")

        # 保存FAISS索引
        for index_name, index in self.indices.items():
            index_path = VECTOR_INDICES_DIR / f"{index_name}_index.faiss"
            faiss.write_index(index, str(index_path))
            logger.info(f"保存索引: {index_path}")

        # 保存元数据
        with open(METADATA_STORE_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.metadata_store, f, indent=2, ensure_ascii=False)
        logger.info(f"保存元数据: {METADATA_STORE_PATH}")

        # 保存索引配置
        index_config = {
            'model_name': self.model_name,
            'dimension': self.dimension,
            'indices': list(self.indices.keys()),
            'creation_time': str(Path.ctime(VECTOR_INDICES_DIR / "main_index.faiss"))
        }

        config_path = VECTOR_INDICES_DIR / "index_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(index_config, f, indent=2)
        logger.info(f"保存索引配置: {config_path}")

    def _print_statistics(self):
        """打印统计信息"""
        print("\n" + "=" * 60)
        print("向量化统计信息:")
        print("-" * 60)

        for index_name, metadata in self.metadata_store.items():
            print(f"\n索引: {index_name}")
            print(f"  类型: {metadata['index_type']}")
            print(f"  向量数: {metadata['total_vectors']}")

            if metadata['index_type'] == 'section':
                print(f"  章节类型: {metadata['section_type']}")

        print("\n" + "=" * 60)
        print(f"索引文件保存在: {VECTOR_INDICES_DIR}")
        print(f"元数据保存在: {METADATA_STORE_PATH}")


def main():
    """主函数"""
    # 自动检测并使用GPU
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vectorizer = EnhancedVectorizer(device=device)
    vectorizer.vectorize_and_index()


if __name__ == "__main__":
    main()