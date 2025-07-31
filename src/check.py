#检查论文数据是否有重复的
import os
import fitz  # PyMuPDF
from thefuzz import fuzz
from tqdm import tqdm
import collections

# --- 配置参数 ---
# PDF文件所在的文件夹路径
PDF_DIRECTORY = r"C:\Users\86152\Desktop\专利"  # <--- 修改这里！

# 标题相似度的阈值 (0-100)。如果两篇论文标题的相似度得分高于此值，则认为它们是重复的。
# 90 通常是一个比较好的起点。
SIMILARITY_THRESHOLD = 90


# --- 函数定义 ---

def extract_title_from_pdf(pdf_path):
    """
    从PDF文件中提取可能的标题。
    这是一个启发式方法：通常标题是文档最开始的几行非空文本。
    """
    try:
        doc = fitz.open(pdf_path)
        # 只读取第一页的文本
        text = doc[0].get_text("text")
        doc.close()

        # 按行分割，并过滤掉空行和过短的行
        lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 5]

        # 假设标题在最开始的5行内，我们将它们拼接起来
        # 这可以处理多行标题的情况
        if lines:
            # 将前几行（最多3行）合并作为标题候选
            potential_title = " ".join(lines[:3])
            return potential_title
        return ""
    except Exception as e:
        print(f"Error processing {os.path.basename(pdf_path)}: {e}")
        return ""


def find_duplicate_papers(directory):
    """
    在指定目录中查找重复的PDF论文。
    """
    pdf_files = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print("No PDF files found in the directory.")
        return

    print(f"Found {len(pdf_files)} PDF files. Extracting titles...")

    # 1. 提取所有论文的标题
    papers_data = []
    for filename in tqdm(pdf_files, desc="Extracting Titles"):
        filepath = os.path.join(directory, filename)
        title = extract_title_from_pdf(filepath)
        if title:
            papers_data.append({"filename": filename, "title": title})

    print(f"\nExtracted titles for {len(papers_data)} papers. Comparing for duplicates...")

    # 2. 比较标题相似度来查找重复项
    duplicates = []
    checked_indices = set()

    for i in tqdm(range(len(papers_data)), desc="Comparing Papers"):
        if i in checked_indices:
            continue

        current_paper = papers_data[i]
        # 初始化一个重复组，至少包含当前论文
        current_group = [current_paper['filename']]

        for j in range(i + 1, len(papers_data)):
            if j in checked_indices:
                continue

            other_paper = papers_data[j]

            # 使用 thefuzz 计算相似度。token_set_ratio 对词序不敏感，效果很好。
            similarity = fuzz.token_set_ratio(current_paper['title'], other_paper['title'])

            if similarity >= SIMILARITY_THRESHOLD:
                current_group.append(other_paper['filename'])
                checked_indices.add(j)

        # 如果一个组里有多于一篇论文，就说明找到了重复
        if len(current_group) > 1:
            duplicates.append(current_group)

        checked_indices.add(i)

    return duplicates


# --- 主程序 ---
if __name__ == "__main__":
    # 确保路径存在
    if not os.path.isdir(PDF_DIRECTORY):
        print(f"Error: Directory not found at '{PDF_DIRECTORY}'")
        print("Please update the 'PDF_DIRECTORY' variable in the script.")
    else:
        duplicate_groups = find_duplicate_papers(PDF_DIRECTORY)

        print("\n--- Duplicate Check Results ---")
        if not duplicate_groups:
            print("No duplicate papers found.")
        else:
            print(f"Found {len(duplicate_groups)} groups of duplicate papers:\n")
            for i, group in enumerate(duplicate_groups):
                print(f"Group {i + 1}:")
                for filename in group:
                    print(f"  - {filename}")
                print("-" * 20)