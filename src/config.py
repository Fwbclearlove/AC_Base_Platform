# src/config.py

import os
from pathlib import Path

# 基础路径配置

BASE_DIR = Path(__file__).parent.parent

# 定义原始数据目录
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

# 定义处理后数据的存放目录
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# 定义提取出的纯文本的存放目录
TEXT_DATA_DIR = PROCESSED_DATA_DIR / "text"
TEXT_DATA_DIR.mkdir(parents=True, exist_ok=True)





# 分块相关配置

# 原始分块文件路径（保留兼容性）
CHUNKS_DATA_PATH = PROCESSED_DATA_DIR / "chunks.json"

# 智能分块文件路径（新增）
SMART_CHUNKS_PATH = PROCESSED_DATA_DIR / "smart_chunks.json"
CHUNKING_STATS_PATH = PROCESSED_DATA_DIR / "chunking_statistics.json"

# 分块策略参数 - 基础设置
CHUNK_SIZE = 1200  # 默认块大小
CHUNK_OVERLAP = 100  # 默认重叠大小

# 分块策略参数 - 按章节类型的细化设置（新增）
SECTION_CHUNK_CONFIGS = {
    'abstract': {
        'chunk_size': 400,
        'overlap': 50
    },
    'introduction': {
        'chunk_size': 600,
        'overlap': 80
    },
    'method': {
        'chunk_size': 500,
        'overlap': 60
    },
    'experiment': {
        'chunk_size': 500,
        'overlap': 60
    },
    'result': {
        'chunk_size': 400,
        'overlap': 50
    },
    'conclusion': {
        'chunk_size': 500,
        'overlap': 60
    },
    'default': {
        'chunk_size': 600,
        'overlap': 80
    }
}

# 最小块大小（低于此大小的块会被合并）
MIN_CHUNK_SIZE = 100



#  RAG系统配置（新增）
# RAG系统目录
RAG_DIR = PROCESSED_DATA_DIR / "rag_system"
RAG_DIR.mkdir(parents=True, exist_ok=True)

# 向量索引目录
VECTOR_INDICES_DIR = RAG_DIR / "vector_indices"
VECTOR_INDICES_DIR.mkdir(parents=True, exist_ok=True)

# 不同类型的索引文件
FAISS_INDEX_PATH = VECTOR_INDICES_DIR / "faiss_index.bin"
CHUNK_ID_MAP_PATH = VECTOR_INDICES_DIR / "chunk_id_map.json"
SUMMARY_INDEX_PATH = VECTOR_INDICES_DIR / "summary_index.bin"
STRUCTURED_INDEX_PATH = VECTOR_INDICES_DIR / "structured_index.bin"

# 元数据存储
METADATA_STORE_PATH = RAG_DIR / "metadata_store.json"



#模型配置
# Embedding模型
EMBEDDING_MODEL = 'BAAI/bge-large-zh-v1.5'
EMBEDDING_DIMENSION = 1024  # BGE-large的维度

# Rerank模型（可选，用于精排）
RERANK_MODEL = 'BAAI/bge-reranker-large'

# 批处理大小
EMBEDDING_BATCH_SIZE = 32  # CPU上的批处理大小
EMBEDDING_BATCH_SIZE_GPU = 64  # GPU上的批处理大小（如果可用）



# 检索配置
# 检索参数
RETRIEVAL_TOP_K = 20  # 初步检索的文档数
RERANK_TOP_K = 5  # 重排后返回的文档数

# 相似度阈值
SIMILARITY_THRESHOLD = 0.7  # 最低相似度阈值




# 其他配置
# 日志配置
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 缓存目录
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)





# 调试模式
DEBUG = False  # 是否开启调试模式
VERBOSE = True  # 是否输出详细信息

# 打印路径以确认是否正确
if __name__ == '__main__':
    print("=" * 60)
    print("配置文件路径检查")
    print("=" * 60)
    print(f"项目根目录: {BASE_DIR}")
    print(f"原始数据目录: {RAW_DATA_DIR}")
    print(f"处理后数据目录: {PROCESSED_DATA_DIR}")
    print(f"纯文本存放目录: {TEXT_DATA_DIR}")
    print("\n新增配置:")
    print(f"智能分块文件: {SMART_CHUNKS_PATH}")
    print(f"RAG系统目录: {RAG_DIR}")
    print(f"向量索引目录: {VECTOR_INDICES_DIR}")
    print(f"Embedding模型: {EMBEDDING_MODEL}")
    print("\n章节分块配置:")
    for section_type, config in SECTION_CHUNK_CONFIGS.items():
        print(f"  {section_type}: size={config['chunk_size']}, overlap={config['overlap']}")