# src/04a_vectorization.py (CPU确认版)
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

from config import PROCESSED_DATA_DIR, CHUNKS_DATA_PATH

# --- 模型和设备配置 ---
# 明确指定使用CPU。后续再在服务器上用Gpu跑
DEVICE = "cpu"
# BGE中英双语大模型
MODEL_NAME = 'BAAI/bge-large-zh-v1.5'
# --- 批处理大小 ---
# 在CPU上，batch_size对性能影响不大，但可以设置一个合理的值
BATCH_SIZE = 32
# --- FAISS索引文件路径 ---
FAISS_INDEX_PATH = PROCESSED_DATA_DIR / "vector_index.faiss"
CHUNK_ID_MAP_PATH = PROCESSED_DATA_DIR / "chunk_id_map.json"
def vectorize_chunks():
    """
    加载分块文本，使用BGE模型进行向量化，并构建FAISS索引。
    """
    # 1. 加载分块数据
    print("加载分块数据...")
    with open(CHUNKS_DATA_PATH, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)

    if not chunks_data:
        print("没有找到任何分块数据，程序退出。")
        return

    chunk_texts = [item['text'] for item in chunks_data]
    chunk_ids = [item['chunk_id'] for item in chunks_data]

    # 2. 加载嵌入模型
    print(f"正在从Hugging Face加载模型: {MODEL_NAME}...")
    print(f"将使用设备: {DEVICE}")  # 这里会打印 "cpu"
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)

    # 3. 批量编码文本，生成向量
    print("开始对所有文本块进行向量化（在CPU上可能需要较长时间）...")
    print(f"使用批处理大小 (Batch Size): {BATCH_SIZE}")

    embeddings = model.encode(chunk_texts,
                              batch_size=BATCH_SIZE,
                              show_progress_bar=True,
                              normalize_embeddings=True)

    print(f"向量化完成！生成了 {len(embeddings)} 个向量。")
    print(f"每个向量的维度是: {embeddings.shape[1]}")

    # 4. 构建并保存FAISS索引
    print("正在构建FAISS索引...")
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)

    index.add(np.array(embeddings, dtype=np.float32))

    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"FAISS索引已保存至: {FAISS_INDEX_PATH}")

    # 5. 保存 chunk_id 到索引位置的映射
    id_map = {i: chunk_id for i, chunk_id in enumerate(chunk_ids)}
    with open(CHUNK_ID_MAP_PATH, 'w', encoding='utf-8') as f:
        json.dump(id_map, f, indent=4)

    print(f"块ID映射文件已保存至: {CHUNK_ID_MAP_PATH}")

if __name__ == "__main__":
    vectorize_chunks()