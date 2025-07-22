import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import uuid
from config import (
    PROCESSED_DATA_DIR,
    TEXT_DATA_DIR,
    CHUNKS_DATA_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)
def chunk_all_texts():
    """
    加载所有已提取的文本，进行分块，并保存结果。
    """
    metadata_path = PROCESSED_DATA_DIR / "metadata.json"

    # 1. 加载元数据文件
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    # 2. 初始化文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""],  # 针对中英文优化的分隔符
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,  # 使用Python的len函数计算长度
    )
    all_chunks = []
    print("开始对所有文本进行分块...")
    # 3. 遍历元数据，处理已提取文本的文件
    for file_id, info in tqdm(metadata.items(), desc="分块进度"):
        if info.get("status") == "text_extracted" and "text_path" in info:
            text_path = info["text_path"]
            try:
                # 4. 读取文本内容
                with open(text_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
                # 5. 执行分块
                chunks = text_splitter.split_text(text_content)
                # 6. 为每个块添加元数据并存入总列表
                for i, chunk_text in enumerate(chunks):
                    chunk_id = f"{file_id}_chunk_{i + 1}"  # 创建唯一的块ID，如: "paper1_chunk_1"
                    all_chunks.append({
                        "chunk_id": chunk_id,
                        "source_id": file_id,  # 块所属的源文件ID
                        "text": chunk_text,
                        "chunk_order": i + 1,  # 块在原文中的顺序
                    })
                # 7. 更新元数据状态
                info["status"] = "chunked"
                info["chunk_count"] = len(chunks)
            except Exception as e:
                print(f"处理文件 {file_id} 时出错: {e}")
                info["status"] = "chunking_failed"
                info["error"] = str(e)
    # 8. 保存所有分块结果到一个大的JSON文件中
    with open(CHUNKS_DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=4, ensure_ascii=False)
    print(f"所有分块已处理完毕！共生成 {len(all_chunks)} 个块。")
    print(f"分块数据已保存至: {CHUNKS_DATA_PATH}")
    # 9. 更新元数据文件
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    print("元数据文件已更新。")
if __name__ == "__main__":
    chunk_all_texts()