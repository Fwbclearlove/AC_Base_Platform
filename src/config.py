# src/config.py

import os
from pathlib import Path

# 获取项目根目录
BASE_DIR = Path(__file__).parent.parent

# 定义原始数据目录
# 将项目根目录和 'data/raw' 拼接起来
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

# 定义处理后数据的存放目录
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

# 确保处理后的数据目录存在，如果不存在则创建
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
# 定义提取出的纯文本的存放目录
TEXT_DATA_DIR = PROCESSED_DATA_DIR / "text"
TEXT_DATA_DIR.mkdir(parents=True, exist_ok=True)
# 定义分块后数据的存放路径
CHUNKS_DATA_PATH = PROCESSED_DATA_DIR / "chunks.json"
# --- 分块策略参数 ---
# 目标块大小（以字符数为单位，可以根据你的文本特性调整）
# 对于学术论文，1000-1500个字符通常能包含几个完整的句子。
CHUNK_SIZE = 1200
# 块之间的重叠大小，确保语义连续性
CHUNK_OVERLAP = 100
# 打印路径以确认是否正确
if __name__ == '__main__':
    print(f"项目根目录: {BASE_DIR}")
    print(f"原始数据目录: {RAW_DATA_DIR}")
    print(f"处理后数据目录: {PROCESSED_DATA_DIR}")
    print(f"纯文本存放目录: {TEXT_DATA_DIR}")