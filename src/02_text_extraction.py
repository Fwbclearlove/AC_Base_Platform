# src/02_text_extraction_basic_reliable.py
import os
import json
import fitz  # PyMuPDF
import re
from tqdm import tqdm
from pathlib import Path

from config import (
    PROCESSED_DATA_DIR,
    TEXT_DATA_DIR
)


class BasicReliablePDFExtractor:
    """基础但可靠的PDF文本提取器 - 专注于不丢失内容"""

    def __init__(self):
        # 只过滤最明显的噪音
        self.obvious_noise_patterns = [
            r'^\d+$',  # 纯数字（页码）
            r'^\s*$',  # 空白
        ]

        # 只过滤最明显的页眉页脚关键词
        self.obvious_header_footer = [
            'www.', 'http', '.com', '.org',
            '©', 'copyright',
        ]

    def is_obvious_noise(self, text):
        """只过滤最明显的噪音，保守策略"""
        if not text or len(text.strip()) < 2:
            return True

        text_clean = text.strip().lower()

        # 只检查最明显的噪音
        for pattern in self.obvious_noise_patterns:
            if re.match(pattern, text_clean):
                return True

        # 只检查最明显的页眉页脚
        for keyword in self.obvious_header_footer:
            if keyword in text_clean and len(text_clean) < 50:  # 短文本中的URL等
                return True

        return False

    def is_likely_page_number_area(self, bbox, page_height, page_width):
        """只过滤最明显的页码区域"""
        x0, y0, x1, y1 = bbox

        # 页面最顶部2%区域
        if y0 < page_height * 0.02:
            return True

        # 页面最底部2%区域
        if y1 > page_height * 0.98:
            return True

        return False

    def basic_clean_text(self, text):
        """基础文本清洗，保守策略"""
        if not text:
            return ""

        # 1. 合并多余空格
        text = re.sub(r'\s+', ' ', text)

        # 2. 修复明显的PDF解析问题
        # 修复断词连字符
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)

        # 3. 清理一些明显的符号问题
        text = text.replace('\\', ' ')  # 移除反斜杠

        # 4. 清理多余的符号（保守）
        text = re.sub(r'\s*\^\s*', ' ', text)  # 清理上标符号

        return text.strip()

    def extract_from_pdf(self, pdf_path, file_id):
        """主提取函数 - 保守策略，重点是不丢失内容"""
        print(f"开始处理: {file_id}")

        try:
            doc = fitz.open(pdf_path)
            all_text_blocks = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                page_height = page.rect.height
                page_width = page.rect.width

                # 获取文本块
                blocks = page.get_text("blocks")

                for block in blocks:
                    if len(block) >= 5:
                        bbox = block[:4]
                        raw_text = block[4]

                        # 只过滤最明显的页码区域
                        if self.is_likely_page_number_area(bbox, page_height, page_width):
                            continue

                        # 只过滤最明显的噪音
                        if self.is_obvious_noise(raw_text):
                            continue

                        # 基础清洗
                        cleaned_text = self.basic_clean_text(raw_text)

                        # 只要有基本内容就保留
                        if cleaned_text and len(cleaned_text) > 5:
                            all_text_blocks.append(cleaned_text)

            doc.close()

            if not all_text_blocks:
                return None, "no_content_extracted"

            # 简单合并所有文本
            final_text = "\n\n".join(all_text_blocks)

            print(f"提取到 {len(all_text_blocks)} 个文本块，总长度: {len(final_text)}")
            return final_text, "text_extracted"

        except Exception as e:
            print(f"处理PDF时出错: {e}")
            return None, "extraction_error"


def extract_all_texts():
    """批量提取所有PDF文本 - 基础可靠版本"""
    metadata_path = PROCESSED_DATA_DIR / "metadata.json"

    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print("错误：metadata.json文件未找到，请先运行01_data_ingestion.py")
        return

    extractor = BasicReliablePDFExtractor()

    print(f"开始基础可靠提取 {len(metadata)} 个PDF文件...")

    processed_count = 0

    for file_id, info in tqdm(metadata.items(), desc="基础可靠提取"):
        source_path = info["source_path"]

        if not os.path.exists(source_path):
            info["status"] = "file_not_found"
            continue

        try:
            extracted_text, status = extractor.extract_from_pdf(source_path, file_id)

            if extracted_text:
                # 保存基础版本文本
                text_file_path = TEXT_DATA_DIR / f"{file_id}_basic.txt"
                with open(text_file_path, 'w', encoding='utf-8') as f:
                    f.write(extracted_text)

                info["status"] = status
                info["text_path"] = str(text_file_path)
                info["text_length"] = len(extracted_text)
                info["extraction_method"] = "basic_reliable"
                info.pop("error", None)

                processed_count += 1
                print(f"成功: {file_id} ({len(extracted_text)} 字符)")
            else:
                info["status"] = status
                info["error"] = "基础提取失败"
                print(f"失败: {file_id}")

        except Exception as e:
            info["status"] = "extraction_error"
            info["error"] = str(e)
            print(f"错误 {file_id}: {e}")

    # 保存元数据
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print(f"\n基础可靠提取完成！")
    print(f"成功处理: {processed_count}/{len(metadata)} 个文件")
    print(f"文件保存格式: *_basic.txt")


if __name__ == "__main__":
    extract_all_texts()