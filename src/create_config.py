# create_config.py
import datetime
import json

from pathlib import Path

config = {
    'model_name': 'BAAI/bge-large-zh-v1.5',
    'dimension': 1024,
    'indices': ['main', 'section_introduction', 'section_abstract',
                'section_related_work', 'section_experiment', 'section_default',
                'section_conclusion', 'section_reference', 'section_method', 'summary'],
    'creation_time': str(datetime.datetime.now())
}

config_path = Path("D:/AC_Base_Platform/data/processed/rag_system/vector_indices/index_config.json")
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
print("配置文件创建成功！")