# src/03_text_chunking_improved.py
import json
import re
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import (
    PROCESSED_DATA_DIR,
    TEXT_DATA_DIR,
    # 使用新的路径配置
    SMART_CHUNKS_PATH,
    CHUNKING_STATS_PATH,
    # 使用新的分块参数
    SECTION_CHUNK_CONFIGS,
    MIN_CHUNK_SIZE,
    # 调试选项
    VERBOSE,
    DEBUG
)


class AcademicTextChunker:
    """学术文档智能分块器"""

    def __init__(self, custom_config: Optional[Dict] = None):
        """
        初始化分块器

        Args:
            custom_config: 自定义配置，覆盖默认配置
        """
        # 章节标题模式
        self.section_patterns = [
            # 英文章节
            r'(?:^|\n)(?:Abstract|ABSTRACT)(?:\n|$)',
            r'(?:^|\n)(?:\d+\.?\s*)?(?:Introduction|INTRODUCTION)(?:\n|$)',
            r'(?:^|\n)(?:\d+\.?\s*)?(?:Related Work|RELATED WORK)(?:\n|$)',
            r'(?:^|\n)(?:\d+\.?\s*)?(?:Method|METHOD|Methods|METHODS|Methodology|METHODOLOGY|Approach|APPROACH)(?:\n|$)',
            r'(?:^|\n)(?:\d+\.?\s*)?(?:Experiment|EXPERIMENT|Experiments|EXPERIMENTS|Evaluation|EVALUATION)(?:\n|$)',
            r'(?:^|\n)(?:\d+\.?\s*)?(?:Result|RESULT|Results|RESULTS)(?:\n|$)',
            r'(?:^|\n)(?:\d+\.?\s*)?(?:Discussion|DISCUSSION)(?:\n|$)',
            r'(?:^|\n)(?:\d+\.?\s*)?(?:Conclusion|CONCLUSION|Conclusions|CONCLUSIONS)(?:\n|$)',
            r'(?:^|\n)(?:\d+\.?\s*)?(?:Reference|REFERENCE|References|REFERENCES)(?:\n|$)',
            r'(?:^|\n)(?:\d+\.?\s*)?(?:Appendix|APPENDIX|Appendices|APPENDICES)(?:\n|$)',

            # 中文章节
            r'(?:^|\n)(?:摘\s*要|摘要|内容摘要)(?:\n|$)',
            r'(?:^|\n)(?:\d+\.?\s*)?(?:引\s*言|引言|绪论|前言|概述)(?:\n|$)',
            r'(?:^|\n)(?:\d+\.?\s*)?(?:相关工作|研究现状|文献综述|研究背景)(?:\n|$)',
            r'(?:^|\n)(?:\d+\.?\s*)?(?:方法|算法|模型|技术路线|研究方法)(?:\n|$)',
            r'(?:^|\n)(?:\d+\.?\s*)?(?:实验|评估|验证|测试|实验分析)(?:\n|$)',
            r'(?:^|\n)(?:\d+\.?\s*)?(?:结果|实验结果|结果分析)(?:\n|$)',
            r'(?:^|\n)(?:\d+\.?\s*)?(?:讨论|分析|结果讨论)(?:\n|$)',
            r'(?:^|\n)(?:\d+\.?\s*)?(?:结论|总结|结语|展望)(?:\n|$)',
            r'(?:^|\n)(?:\d+\.?\s*)?(?:参考文献|引用文献)(?:\n|$)',
            r'(?:^|\n)(?:\d+\.?\s*)?(?:附录|附件)(?:\n|$)',

            # 专利文档章节
            r'(?:^|\n)(?:技术领域|Technical Field|TECHNICAL FIELD)(?:\n|$)',
            r'(?:^|\n)(?:背景技术|Background|BACKGROUND|Prior Art|PRIOR ART)(?:\n|$)',
            r'(?:^|\n)(?:发明内容|Summary of Invention|SUMMARY)(?:\n|$)',
            r'(?:^|\n)(?:具体实施方式|Detailed Description|DETAILED DESCRIPTION|实施例|Embodiment)(?:\n|$)',
            r'(?:^|\n)(?:权利要求|Claims|CLAIMS)(?:\n|$)',
            r'(?:^|\n)(?:说明书附图|Description of Drawings|DRAWINGS)(?:\n|$)',
        ]

        # 从配置文件加载分块参数
        self.section_configs = custom_config if custom_config else SECTION_CHUNK_CONFIGS
        self.min_chunk_size = MIN_CHUNK_SIZE
        self.verbose = VERBOSE
        self.debug = DEBUG

        # 添加章节类型映射缓存
        self._section_type_cache = {}

    def extract_sections(self, text: str) -> List[Tuple[str, str, int]]:
        """
        提取文档的章节
        返回: [(章节标题, 章节内容, 起始位置)]
        """
        sections = []
        section_matches = []

        # 找到所有章节标题
        for pattern in self.section_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                section_title = match.group().strip()
                # 标准化章节标题（去除多余空格和编号）
                section_title = re.sub(r'^\d+\.?\s*', '', section_title)
                section_title = ' '.join(section_title.split())

                section_matches.append({
                    'title': section_title,
                    'start': match.start(),
                    'end': match.end()
                })

        if not section_matches:
            # 没有找到章节，返回整个文档
            return [('全文', text, 0)]

        # 按位置排序
        section_matches.sort(key=lambda x: x['start'])

        # 去除重复的章节匹配（位置太接近的）
        filtered_matches = []
        for match in section_matches:
            if not filtered_matches or match['start'] - filtered_matches[-1]['end'] > 10:
                filtered_matches.append(match)
        section_matches = filtered_matches

        # 提取每个章节的内容
        for i, section in enumerate(section_matches):
            title = section['title']
            start = section['end']

            # 确定章节结束位置
            if i + 1 < len(section_matches):
                end = section_matches[i + 1]['start']
            else:
                end = len(text)

            content = text[start:end].strip()

            # 过滤太短的章节（可能是误匹配）
            if len(content) > 50:
                sections.append((title, content, section['start']))

        # 如果第一个章节不是从头开始，添加一个"前言"章节
        if sections and sections[0][2] > 100:
            preface = text[:sections[0][2]].strip()
            if len(preface) > 50:
                sections.insert(0, ('前言', preface, 0))

        return sections

    def determine_section_type(self, section_title: str) -> str:
        """根据章节标题确定章节类型"""
        # 使用缓存提高性能
        if section_title in self._section_type_cache:
            return self._section_type_cache[section_title]

        title_lower = section_title.lower()

        # 移除可能的编号
        title_lower = re.sub(r'^\d+\.?\s*', '', title_lower)

        section_type = 'default'

        if any(word in title_lower for word in ['abstract', '摘要', '内容摘要']):
            section_type = 'abstract'
        elif any(word in title_lower for word in ['introduction', '引言', '绪论', '前言', '概述']):
            section_type = 'introduction'
        elif any(word in title_lower for word in ['method', '方法', '算法', '模型', 'approach', '技术路线']):
            section_type = 'method'
        elif any(word in title_lower for word in ['experiment', '实验', '评估', 'evaluation', '验证', '测试']):
            section_type = 'experiment'
        elif any(word in title_lower for word in ['result', '结果']):
            section_type = 'result'
        elif any(word in title_lower for word in ['conclusion', '结论', '总结', '结语']):
            section_type = 'conclusion'
        elif any(word in title_lower for word in ['reference', '参考文献', '引用文献']):
            section_type = 'reference'
        elif any(word in title_lower for word in ['related', '相关工作', '研究现状', '文献综述']):
            section_type = 'related_work'
        elif any(word in title_lower for word in ['discussion', '讨论', '分析']):
            section_type = 'discussion'

        # 缓存结果
        self._section_type_cache[section_title] = section_type
        return section_type

    def chunk_section(self, section_text: str, section_type: str) -> List[str]:
        """
        对章节内容进行智能分块

        Args:
            section_text: 章节文本
            section_type: 章节类型
        """
        # 从配置获取参数
        config = self.section_configs.get(section_type, self.section_configs['default'])
        chunk_size = config['chunk_size']
        chunk_overlap = config['overlap']

        # 根据章节类型选择分割符
        if section_type in ['method', 'experiment']:
            # 方法和实验部分优先按步骤、编号分割
            separators = [
                "\n\n",  # 段落
                "\n",  # 换行
                r"\d+\.",  # 编号 1. 2. 3.
                r"\(\d+\)",  # (1) (2) (3)
                r"[A-Z]\.",  # A. B. C.
                r"\([a-z]\)",  # (a) (b) (c)
                "；",  # 中文分号
                ";",  # 英文分号
                "。",  # 中文句号
                ".",  # 英文句号
                "，",  # 中文逗号
                ",",  # 英文逗号
                " ",  # 空格
                ""
            ]
        elif section_type == 'abstract':
            # 摘要尽量保持完整
            separators = ["\n\n", "\n", "。", ".", " ", ""]
        elif section_type == 'reference':
            # 参考文献按条目分割
            separators = [
                r"\[\d+\]",  # [1] [2] 格式
                r"\d+\.",  # 1. 2. 格式
                "\n\n",
                "\n",
                ""
            ]
        else:
            # 其他章节的默认分割
            separators = [
                "\n\n",
                "\n",
                "。",
                ".",
                "！",
                "!",
                "？",
                "?",
                "；",
                ";",
                "，",
                ",",
                " ",
                ""
            ]

        # 创建文本分割器
        text_splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=True  # 支持正则表达式分隔符
        )

        # 执行分块
        chunks = text_splitter.split_text(section_text)

        # 后处理：确保每个块都有足够的内容
        processed_chunks = []

        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue

            if len(chunk) < self.min_chunk_size and processed_chunks:
                # 太短的块合并到前一个块
                processed_chunks[-1] += "\n" + chunk
            else:
                processed_chunks.append(chunk)

        # 如果处理后没有任何块，至少返回原文
        if not processed_chunks and section_text.strip():
            processed_chunks = [section_text.strip()]

        return processed_chunks

    def smart_chunk_document(self, text: str, file_id: str,
                             metadata: Dict = None) -> List[Dict]:
        """
        对文档进行智能分块

        Returns:
            分块列表，每个块包含详细元数据
        """
        all_chunks = []

        # 1. 提取章节
        sections = self.extract_sections(text)

        if self.verbose:
            print(f"  文档 {file_id} 识别到 {len(sections)} 个章节")

        # 2. 对每个章节进行分块
        global_chunk_index = 0  # 全局块索引

        for section_idx, (section_title, section_text, section_pos) in enumerate(sections):
            section_type = self.determine_section_type(section_title)

            # 章节级别的分块
            section_chunks = self.chunk_section(section_text, section_type)

            if self.verbose and len(section_chunks) > 0:
                print(f"    章节 '{section_title}' ({section_type}) -> {len(section_chunks)} 个块")

            # 3. 为每个块添加丰富的元数据
            for chunk_idx, chunk_text in enumerate(section_chunks):
                chunk_data = {
                    # 基础标识
                    'chunk_id': f"{file_id}_s{section_idx}_c{chunk_idx}",
                    'source_id': file_id,
                    'text': chunk_text,

                    # 章节信息
                    'section_title': section_title,
                    'section_type': section_type,
                    'section_index': section_idx,
                    'total_sections': len(sections),

                    # 块位置信息（局部和全局）
                    'chunk_index': global_chunk_index,  # 全局索引
                    'chunk_index_in_section': chunk_idx,  # 章节内索引
                    'total_chunks_in_section': len(section_chunks),

                    # 上下文信息
                    'has_previous': chunk_idx > 0,
                    'has_next': chunk_idx < len(section_chunks) - 1,
                    'is_section_start': chunk_idx == 0,
                    'is_section_end': chunk_idx == len(section_chunks) - 1,

                    # 内容特征
                    'chunk_length': len(chunk_text),
                    'chunk_method': 'hierarchical',  # 标记使用的分块方法

                    # 用于检索的权重提示（可以在检索时使用）
                    'importance_weight': self._calculate_importance_weight(section_type, chunk_idx, len(section_chunks))
                }

                # 添加额外的元数据
                if metadata:
                    chunk_data['document_metadata'] = metadata

                all_chunks.append(chunk_data)
                global_chunk_index += 1

        return all_chunks

    def _calculate_importance_weight(self, section_type: str, chunk_idx: int, total_chunks: int) -> float:
        """
        计算块的重要性权重（用于检索时的排序）

        Args:
            section_type: 章节类型
            chunk_idx: 块在章节中的索引
            total_chunks: 章节总块数

        Returns:
            重要性权重 (0.0 - 1.0)
        """
        # 基础权重
        base_weights = {
            'abstract': 0.9,
            'introduction': 0.8,
            'method': 0.85,
            'experiment': 0.85,
            'result': 0.9,
            'conclusion': 0.85,
            'discussion': 0.75,
            'related_work': 0.6,
            'reference': 0.4,
            'default': 0.7
        }

        base_weight = base_weights.get(section_type, 0.7)

        # 位置调整（开头和结尾的块通常更重要）
        position_factor = 1.0
        if chunk_idx == 0:  # 章节开头
            position_factor = 1.1
        elif chunk_idx == total_chunks - 1:  # 章节结尾
            position_factor = 1.05

        return min(base_weight * position_factor, 1.0)


def chunk_all_texts_improved():
    """
    改进的文本分块主函数
    """
    metadata_path = PROCESSED_DATA_DIR / "metadata.json"

    # 1. 加载元数据
    print("加载元数据...")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # 2. 初始化智能分块器
    chunker = AcademicTextChunker()

    all_chunks = []
    chunk_statistics = {
        'total_documents': 0,
        'total_chunks': 0,
        'chunks_by_section': {},
        'avg_chunk_length': 0,
        'failed_documents': [],
        'text_sources': {'cleaned': 0, 'extracted': 0}
    }

    print("\n开始智能分块处理...")
    print("=" * 60)

    # 3. 优先处理清洗后的文本，如果没有则使用原始提取的文本
    for file_id, info in tqdm(metadata.items(), desc="文档分块进度"):
        text_content = None
        text_source = None

        # 优先使用清洗后的文本
        if info.get("cleaning_status") == "cleaned" and "cleaned_text_path" in info:
            text_path = info["cleaned_text_path"]
            text_source = "cleaned"
        # 否则使用原始提取的文本
        elif info.get("status") == "text_extracted" and "text_path" in info:
            text_path = info["text_path"]
            text_source = "extracted"
        else:
            if VERBOSE:
                print(f"  跳过 {file_id}：没有可用的文本")
            continue

        try:
            # 读取文本
            with open(text_path, 'r', encoding='utf-8') as f:
                text_content = f.read()

            # 跳过太短的文档
            if len(text_content.strip()) < 100:
                if VERBOSE:
                    print(f"  跳过 {file_id}：文本太短 ({len(text_content)} 字符)")
                continue

            # 准备文档元数据（可以从概要层获取更多信息）
            doc_metadata = {
                'text_source': text_source,
                'original_status': info.get("status"),
                'document_type': _infer_document_type(file_id, info)
            }

            # 执行智能分块
            chunks = chunker.smart_chunk_document(
                text_content,
                file_id,
                metadata=doc_metadata
            )

            if chunks:
                # 添加到总列表
                all_chunks.extend(chunks)

                # 更新统计
                chunk_statistics['total_documents'] += 1
                chunk_statistics['total_chunks'] += len(chunks)
                chunk_statistics['text_sources'][text_source] += 1

                # 计算平均块长度
                total_length = sum(chunk['chunk_length'] for chunk in chunks)
                chunk_statistics['avg_chunk_length'] = (
                        (chunk_statistics['avg_chunk_length'] * (
                                    chunk_statistics['total_chunks'] - len(chunks)) + total_length)
                        / chunk_statistics['total_chunks']
                )

                # 统计各章节的块数
                for chunk in chunks:
                    section_type = chunk['section_type']
                    if section_type not in chunk_statistics['chunks_by_section']:
                        chunk_statistics['chunks_by_section'][section_type] = 0
                    chunk_statistics['chunks_by_section'][section_type] += 1

                # 更新元数据
                info["chunking_status"] = "smart_chunked"
                info["chunk_count"] = len(chunks)
                info["chunk_source"] = text_source
                info["avg_chunk_length"] = total_length / len(chunks) if chunks else 0

        except Exception as e:
            print(f"  处理文件 {file_id} 时出错: {e}")
            info["chunking_status"] = "failed"
            info["chunking_error"] = str(e)
            chunk_statistics['failed_documents'].append(file_id)

            if DEBUG:
                import traceback
                traceback.print_exc()

    # 4. 保存分块结果
    print("\n保存分块结果...")

    # 保存所有分块
    with open(SMART_CHUNKS_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    # 保存统计信息
    with open(CHUNKING_STATS_PATH, 'w', encoding='utf-8') as f:
        json.dump(chunk_statistics, f, indent=2, ensure_ascii=False)

    # 更新元数据
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # 5. 打印统计结果
    print("\n" + "=" * 60)
    print("智能分块完成！")
    print(f"处理文档数: {chunk_statistics['total_documents']}")
    print(f"生成块总数: {chunk_statistics['total_chunks']}")
    if chunk_statistics['total_documents'] > 0:
        print(f"平均每文档块数: {chunk_statistics['total_chunks'] / chunk_statistics['total_documents']:.1f}")
        print(f"平均块长度: {chunk_statistics['avg_chunk_length']:.0f} 字符")

    print(f"\n文本来源分布:")
    print(f"  清洗文本: {chunk_statistics['text_sources']['cleaned']} 个文档")
    print(f"  原始文本: {chunk_statistics['text_sources']['extracted']} 个文档")

    print("\n各章节类型的块数分布:")
    sorted_sections = sorted(chunk_statistics['chunks_by_section'].items(), key=lambda x: x[1], reverse=True)
    for section_type, count in sorted_sections:
        percentage = (count / chunk_statistics['total_chunks'] * 100) if chunk_statistics['total_chunks'] > 0 else 0
        print(f"  {section_type:15s}: {count:4d} 块 ({percentage:5.1f}%)")

    if chunk_statistics['failed_documents']:
        print(f"\n失败文档数: {len(chunk_statistics['failed_documents'])}")
        if VERBOSE:
            print("失败文档列表:", chunk_statistics['failed_documents'][:5])

    print(f"\n分块数据已保存至: {SMART_CHUNKS_PATH}")
    print(f"统计信息已保存至: {CHUNKING_STATS_PATH}")
    print(f"元数据已更新: {metadata_path}")


def _infer_document_type(file_id: str, metadata: Dict) -> str:
    """
    推断文档类型

    Args:
        file_id: 文件ID
        metadata: 文档元数据

    Returns:
        文档类型字符串
    """
    # 尝试从元数据中获取
    if 'document_type' in metadata:
        return metadata['document_type']

    # 尝试从概要信息推断
    summaries_path = PROCESSED_DATA_DIR / "document_summaries.json"
    if summaries_path.exists():
        try:
            with open(summaries_path, 'r', encoding='utf-8') as f:
                summaries = json.load(f)
                if file_id in summaries:
                    return summaries[file_id].get('document_type', 'unknown')
        except:
            pass

    # 从文件名推断
    file_id_lower = file_id.lower()
    if 'patent' in file_id_lower or 'cn' in file_id_lower or 'us' in file_id_lower:
        return 'patent'
    elif 'paper' in file_id_lower or 'article' in file_id_lower:
        return 'academic_paper'

    return 'unknown'


if __name__ == "__main__":
    chunk_all_texts_improved()