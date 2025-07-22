# src/01_data_ingestion.py
import os
import json
from tqdm import tqdm
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR
def find_pdf_files(root_dir):
    """
    递归扫描指定目录及其所有子目录，找出所有PDF文件的路径。
    Args:
        root_dir (str or Path): 要扫描的根目录。
    Returns:
        list: 包含所有找到的PDF文件绝对路径的列表。
    """
    pdf_files = []
    print(f"开始扫描目录: {root_dir}")

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".pdf"):
                full_path = os.path.abspath(os.path.join(dirpath, filename))
                pdf_files.append(full_path)

    print(f"扫描完成！共找到 {len(pdf_files)} 个PDF文件。")
    return pdf_files

def create_metadata_file(pdf_list, output_path):
    """
    为找到的PDF文件列表创建一个元数据文件。
    Args:
        pdf_list (list): PDF文件路径列表。
        output_path (str or Path): 元数据文件的输出路径。
    """
    metadata = {}
    print("正在为每个文件创建元数据...")

    # 使用tqdm来显示进度条
    for filepath in tqdm(pdf_list, desc="处理文件"):
        # 使用文件名（不含扩展名）作为唯一ID
        file_id = os.path.splitext(os.path.basename(filepath))[0]

        # 简单地清洗一下ID，替换空格等特殊字符
        file_id_clean = file_id.replace(" ", "_").replace("-", "_")

        metadata[file_id_clean] = {
            "id": file_id_clean,
            "source_path": filepath,
            "status": "unprocessed"  # 标记为“未处理”状态
        }

    # 将元数据以JSON格式写入文件，方便后续程序读取
    # indent=4 让JSON文件格式更美观，易于阅读
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    print(f"元数据文件已成功创建于: {output_path}")


if __name__ == "__main__":
    # 1. 扫描并获取所有PDF文件列表
    all_pdf_files = find_pdf_files(RAW_DATA_DIR)
    if all_pdf_files:
        # 2. 定义元数据文件的存放位置
        metadata_filepath = PROCESSED_DATA_DIR / "metadata.json"
        # 3. 创建元数据文件
        create_metadata_file(all_pdf_files, metadata_filepath)
    else:
        print("在指定的目录中没有找到任何PDF文件，请检查路径和文件。")