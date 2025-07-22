import fitz
import json
import re
from tqdm import tqdm
from config import PROCESSED_DATA_DIR, TEXT_DATA_DIR
def clean_text(text):
    """
    对提取的文本进行初步清洗。
    - 替换多个换行符为一个
    - 移除非法或不需要的字符（可以根据需要扩展）
    - 去除每行前后的多余空格
    """
    # 将多个连续的换行符替换为单个，以减少空行
    text = re.sub(r'\n\s*\n', '\n', text)
    # 去除每行开头和结尾的空格或制表符
    text = "\n".join([line.strip() for line in text.split('\n')])
    return text


def extract_text_from_pdf(pdf_path):
    """
    使用PyMuPDF从单个PDF文件中提取所有文本。
    Args:
        pdf_path (str): PDF文件的路径。
    Returns:
        str: 包含PDF所有页面文本的字符串，如果出错则返回None。
    """
    try:
        doc = fitz.open(pdf_path)
        full_text = []
        for page in doc:
            full_text.append(page.get_text())
        doc.close()
        return "".join(full_text)
    except Exception as e:
        print(f"错误：无法处理文件 {pdf_path}. 原因: {e}")
        return None


def process_all_pdfs():
    """
    主处理函数：加载元数据，提取文本，并更新元数据。
    """
    metadata_path = PROCESSED_DATA_DIR / "metadata.json"
    # 1. 加载元数据文件
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    print("开始处理PDF文件，提取纯文本...")
    # 使用tqdm来显示总体处理进度
    for file_id, info in tqdm(metadata.items(), desc="总体进度"):
        # 2. 检查文件状态，只处理未被处理过的
        if info["status"] == "unprocessed":
            source_path = info["source_path"]
            # 3. 提取文本
            extracted_text = extract_text_from_pdf(source_path)
            if extracted_text:
                # 4. 清洗文本
                cleaned_text = clean_text(extracted_text)
                # 5. 定义并保存文本文件
                text_filename = f"{file_id}.txt"
                text_filepath = TEXT_DATA_DIR / text_filename
                with open(text_filepath, 'w', encoding='utf-8') as text_file:
                    text_file.write(cleaned_text)
                # 6. 更新元数据
                info["status"] = "text_extracted"
                info["text_path"] = str(text_filepath.resolve())  # 存储绝对路径
                info["error"] = None  # 清除可能存在的旧错误信息
            else:
                # 如果提取失败，也在元数据中记录下来
                info["status"] = "extraction_failed"
                info["error"] = f"Failed to extract text from {source_path}"
    # 7. 将更新后的元数据写回文件
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    print("所有PDF文本提取和清洗完成！元数据已更新。")
if __name__ == "__main__":
    process_all_pdfs()