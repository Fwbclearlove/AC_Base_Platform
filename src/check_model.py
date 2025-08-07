# check_model_cache.py
import os
from pathlib import Path

cache_base = Path(r"C:\Users\86152\.cache\huggingface\hub\models--BAAI--bge-large-zh-v1.5")

# 查找snapshots
if cache_base.exists():
    print("缓存目录结构:")
    for item in cache_base.rglob("*"):
        if item.is_file():
            print(f"  {item.relative_to(cache_base)}")

    # 查找具体的模型文件
    model_files = list(cache_base.rglob("*.bin")) + list(cache_base.rglob("*.safetensors"))
    if model_files:
        print(f"\n找到模型文件: {model_files[0].parent}")