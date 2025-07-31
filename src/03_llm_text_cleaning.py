# src/03_llm_text_cleaning_improved.py
import os
import json
import time
from zhipuai import ZhipuAI
from tqdm import tqdm
from pathlib import Path

from config import PROCESSED_DATA_DIR, TEXT_DATA_DIR

# 使用你们已有的API配置
API_KEY = "a5528da8417645d8a4dfb71a9d30f140.7i6b5zfJDRYLgE9C"
CLIENT = ZhipuAI(api_key=API_KEY)
MODEL_NAME = "glm-4"

# 清洗后文本保存路径
CLEANED_TEXT_DIR = PROCESSED_DATA_DIR / "cleaned_text"
CLEANED_TEXT_DIR.mkdir(exist_ok=True)

# 改进的提示词
CLEANING_PROMPT_TEMPLATE = """
你正在清洗学术论文的第{chunk_idx}部分（共{total_chunks}部分）。

重要指示：
- 这只是论文的一个片段，不是完整论文
- 请只清洗当前片段的内容，不要生成完整的论文结构
- 不要添加不存在的章节标题（如Abstract、Introduction等）
- 只去除噪音，保留核心学术内容

需要清理的内容：
- 页眉、页脚、页码
- 重复信息和乱码
- PDF解析错误
- 多余的符号和格式问题

需要保留的内容：
- 当前片段的所有学术内容
- 技术术语和数据
- 原有的逻辑结构
- 公式和图表说明

第{chunk_idx}部分内容：
---
{raw_text}
---

请直接输出清洗后的片段内容（不要添加任何解释）：
"""


class LLMTextCleaner:
    """使用大模型进行文本清洗"""

    def __init__(self):
        self.max_chunk_size = 8000  # GLM-4 token限制考虑
        self.retry_count = 3
        self.success_count = 0
        self.total_count = 0

    def split_text_for_llm(self, text):
        """将长文本分割成适合LLM处理的块"""
        if len(text) <= self.max_chunk_size:
            return [text]

        # 简单按段落分割
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk + para) <= self.max_chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def clean_text_with_llm(self, text_chunk, file_id, chunk_idx=0, total_chunks=1):
        """使用LLM清洗单个文本块"""
        for attempt in range(self.retry_count):
            try:
                response = CLIENT.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {
                            "role": "user",
                            "content": CLEANING_PROMPT_TEMPLATE.format(
                                raw_text=text_chunk,
                                chunk_idx=chunk_idx + 1,
                                total_chunks=total_chunks
                            )
                        }
                    ],
                    temperature=0.1,  # 低温度确保稳定输出
                    max_tokens=4000,
                )

                cleaned_text = response.choices[0].message.content.strip()

                if cleaned_text:
                    print(f"成功清洗: {file_id} chunk_{chunk_idx + 1}")
                    return cleaned_text
                else:
                    print(f"空返回: {file_id} chunk_{chunk_idx + 1}, 尝试 {attempt + 1}")

            except Exception as e:
                print(f"API调用失败: {file_id} chunk_{chunk_idx + 1}, 尝试 {attempt + 1}, 错误: {e}")
                time.sleep(5)  # 等待后重试

        print(f"最终失败: {file_id} chunk_{chunk_idx + 1}")
        return None

    def clean_single_file(self, file_path, file_id):
        """清洗单个文件"""
        try:
            # 读取原始文本
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()

            if not raw_text or len(raw_text.strip()) < 100:
                print(f"文件内容过短，跳过: {file_id}")
                return None, "content_too_short"

            print(f"开始清洗: {file_id} (原始长度: {len(raw_text)})")

            # 分割文本
            text_chunks = self.split_text_for_llm(raw_text)
            total_chunks = len(text_chunks)
            print(f"分割为 {total_chunks} 个块")

            # 清洗每个块
            cleaned_chunks = []
            for i, chunk in enumerate(text_chunks):
                cleaned = self.clean_text_with_llm(chunk, file_id, i, total_chunks)
                if cleaned:
                    cleaned_chunks.append(cleaned)
                else:
                    # 如果清洗失败，保留原始文本
                    print(f"清洗失败，保留原文: {file_id} chunk_{i + 1}")
                    cleaned_chunks.append(chunk)

                # API调用间隔
                time.sleep(2)

            # 合并清洗后的文本
            final_cleaned_text = "\n\n".join(cleaned_chunks)

            print(f"完成清洗: {file_id} (清洗后长度: {len(final_cleaned_text)})")
            return final_cleaned_text, "cleaned"

        except Exception as e:
            print(f"处理文件错误: {file_id}, {e}")
            return None, "processing_error"


def run_single_file_cleaning():
    """清洗单个文件"""
    metadata_path = PROCESSED_DATA_DIR / "metadata.json"

    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print("错误：metadata.json文件未找到")
        return

    # 找到第一个可用的基础提取文件
    basic_files = {k: v for k, v in metadata.items()
                   if v.get("extraction_method") == "basic_reliable"
                   and v.get("status") == "text_extracted"}

    if not basic_files:
        print("没有找到可处理的基础提取文件")
        return

    # 选择第一个文件进行测试
    file_id = list(basic_files.keys())[0]
    info = basic_files[file_id]

    basic_file_path = info.get("text_path")
    if not basic_file_path or not os.path.exists(basic_file_path):
        print(f"错误：基础文件不存在 {basic_file_path}")
        return

    cleaner = LLMTextCleaner()

    print(f"开始清洗文件: {file_id}")
    print("=" * 50)

    try:
        # 执行清洗
        cleaned_text, status = cleaner.clean_single_file(basic_file_path, file_id)

        if cleaned_text:
            # 保存结果
            result_file_path = CLEANED_TEXT_DIR / f"{file_id}_improved_cleaned.txt"
            with open(result_file_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)

            print("=" * 50)
            print(f"单文件清洗完成！")
            print(f"原始文件: {basic_file_path}")
            print(f"清洗结果: {result_file_path}")
            print(f"原始长度: {len(open(basic_file_path, 'r', encoding='utf-8').read())}")
            print(f"清洗后长度: {len(cleaned_text)}")
            print(f"状态: {status}")

            # 显示前500字符预览
            print("\n清洗结果预览（前500字符）:")
            print("-" * 30)
            print(cleaned_text[:500])
            print("-" * 30)

        else:
            print(f"清洗失败: {status}")

    except Exception as e:
        print(f"清洗异常: {e}")


def run_batch_cleaning():
    """批量清洗所有文件"""
    metadata_path = PROCESSED_DATA_DIR / "metadata.json"

    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print("错误：请先运行基础文本提取")
        return

    cleaner = LLMTextCleaner()

    # 找到所有基础提取的文件
    basic_files = {k: v for k, v in metadata.items()
                   if v.get("extraction_method") == "basic_reliable"
                   and v.get("status") == "text_extracted"}

    if not basic_files:
        print("没有找到基础提取的文件，请先运行 02_text_extraction_basic_reliable.py")
        return

    print(f"开始LLM清洗 {len(basic_files)} 个文件...")

    for file_id, info in tqdm(basic_files.items(), desc="LLM文本清洗"):
        # 检查是否已经清洗过
        if info.get("cleaning_status") == "cleaned":
            print(f"跳过已清洗: {file_id}")
            continue

        basic_file_path = info.get("text_path")
        if not basic_file_path or not os.path.exists(basic_file_path):
            print(f"基础文件不存在: {file_id}")
            continue

        try:
            # LLM清洗
            cleaned_text, status = cleaner.clean_single_file(basic_file_path, file_id)

            if cleaned_text:
                # 保存清洗后的文本
                cleaned_file_path = CLEANED_TEXT_DIR / f"{file_id}_cleaned.txt"
                with open(cleaned_file_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_text)

                # 更新元数据
                info["cleaning_status"] = status
                info["cleaned_text_path"] = str(cleaned_file_path)
                info["cleaned_text_length"] = len(cleaned_text)
                info.pop("cleaning_error", None)

                cleaner.success_count += 1
                print(f"清洗完成: {file_id} (状态: {status})")
            else:
                info["cleaning_status"] = status
                info["cleaning_error"] = "LLM清洗失败"
                print(f"清洗失败: {file_id}")

        except Exception as e:
            info["cleaning_status"] = "cleaning_error"
            info["cleaning_error"] = str(e)
            print(f"处理异常: {file_id}, {e}")

        cleaner.total_count += 1

        # 每处理5个文件保存一次
        if cleaner.total_count % 5 == 0:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
            print(f"中间保存... 已处理 {cleaner.total_count}")

    # 最终保存
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print(f"\nLLM文本清洗完成！")
    print(f"成功清洗: {cleaner.success_count}/{cleaner.total_count}")
    print(f"清洗文件保存在: {CLEANED_TEXT_DIR}")
    print(f"文件格式: *_cleaned.txt")


def main():
    """主程序入口"""
    print("=" * 60)
    print("LLM文本清洗程序")
    print("=" * 60)
    print("请选择操作:")
    print("1. 清洗单个文件（测试用）")
    print("2. 批量清洗所有文件")
    print("0. 退出")
    print("=" * 60)

    while True:
        try:
            choice = input("请输入选择 (0/1/2): ").strip()

            if choice == "0":
                print("程序退出")
                break
            elif choice == "1":
                print("\n选择：清洗单个文件")
                run_single_file_cleaning()
                break
            elif choice == "2":
                print("\n选择：批量清洗所有文件")
                confirm = input("确定要批量处理所有文件吗? (y/n): ").strip().lower()
                if confirm == 'y':
                    run_batch_cleaning()
                else:
                    print("取消批量处理")
                break
            else:
                print("无效选择，请输入 0、1 或 2")

        except KeyboardInterrupt:
            print("\n程序被中断")
            break
        except Exception as e:
            print(f"输入错误: {e}")


if __name__ == "__main__":
    main()