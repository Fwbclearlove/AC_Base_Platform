# src/test_vector_search.py

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import time

from config import (
    PROCESSED_DATA_DIR,
    CHUNKS_DATA_PATH
)

# --- 配置 ---
# 确保使用与向量化时完全相同的模型！
MODEL_NAME = 'BAAI/bge-large-zh-v1.5'
DEVICE = "cpu"  # 在CPU上进行测试即可

# FAISS索引和映射文件的路径
FAISS_INDEX_PATH = PROCESSED_DATA_DIR / "vector_index.faiss"
CHUNK_ID_MAP_PATH = PROCESSED_DATA_DIR / "chunk_id_map.json"


# --- 加载模块 ---
class VectorSearcher:
    def __init__(self):
        print("正在初始化向量搜索引擎...")

        # 1. 加载FAISS索引
        print(f"加载FAISS索引从: {FAISS_INDEX_PATH}")
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))
        print(f"索引加载成功，包含 {self.index.ntotal} 个向量。")

        # 2. 加载ID映射
        print(f"加载块ID映射从: {CHUNK_ID_MAP_PATH}")
        with open(CHUNK_ID_MAP_PATH, 'r', encoding='utf-8') as f:
            self.id_map = json.load(f)

        # 3. 加载原始分块数据，用于显示结果
        print(f"加载原始分块数据从: {CHUNKS_DATA_PATH}")
        with open(CHUNKS_DATA_PATH, 'r', encoding='utf-8') as f:
            self.chunks_data = {item['chunk_id']: item for item in json.load(f)}

        # 4. 加载BGE模型
        print(f"加载嵌入模型: {MODEL_NAME} (这可能需要一些时间)...")
        self.model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        print("模型加载完毕。搜索引擎初始化完成！\n")

    def search(self, query, k=3):
        """
        接收一个查询文本，返回最相关的k个块。
        """
        if not query:
            print("查询不能为空！")
            return

        print(f"--- 正在搜索: '{query}' ---")
        start_time = time.time()

        # 1. 将查询文本向量化
        query_vector = self.model.encode(query, normalize_embeddings=True)
        # FAISS需要一个二维数组作为输入
        query_vector = np.array([query_vector], dtype=np.float32)

        # 2. 在FAISS索引中执行搜索
        # D: 距离（对于归一化向量的内积，越大越好）
        # I: 索引ID (FAISS内部的整数索引)
        distances, indices = self.index.search(query_vector, k)

        end_time = time.time()
        print(f"搜索耗时: {end_time - start_time:.4f} 秒")

        # 3. 解析并打印结果
        print(f"\n找到 Top {k} 个最相关的结果：")
        for i in range(k):
            faiss_index = indices[0][i]
            distance = distances[0][i]

            # 使用ID映射，将FAISS索引转换为我们自己的chunk_id
            chunk_id = self.id_map.get(str(faiss_index))

            if not chunk_id:
                print(f"\n--- 结果 {i + 1} ---")
                print("错误：在id_map中找不到对应的chunk_id！")
                continue

            # 从原始数据中获取块的详细信息
            chunk_info = self.chunks_data.get(chunk_id)

            print(f"\n--- 结果 {i + 1} | 相似度: {distance:.4f} ---")
            print(f"来源文件ID: {chunk_info['source_id']}")
            print(f"块ID: {chunk_id}")
            print("-" * 20)
            print(chunk_info['text'])
            print("-" * (len(chunk_info['text']) if len(chunk_info['text']) > 20 else 20))


# --- 主测试逻辑 ---
if __name__ == "__main__":
    searcher = VectorSearcher()

    # 定义一些测试用例
    test_queries = [
        "什么是SKFD-Isomap？",  # 精确概念查询
        "这篇论文使用了什么分类器？",  # 细节查询
        "流形学习在人脸识别中的应用",  # 模糊主题查询
        "introduction",  # 单个关键词查询
        "有什么方法可以处理非线性问题？"  # 开放式问题查询
    ]

    # 自动执行预设的测试用例
    for q in test_queries:
        searcher.search(q, k=3)
        print("\n" + "=" * 80 + "\n")

    # 提供交互式查询模式
    print("--- 进入交互式查询模式 ---")
    print("输入 'quit' 或 'exit' 退出。")
    while True:
        user_query = input("请输入你的查询: ")
        if user_query.lower() in ['quit', 'exit']:
            break
        searcher.search(user_query, k=3)