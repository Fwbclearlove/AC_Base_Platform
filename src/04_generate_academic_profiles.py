# src/04_generate_academic_profiles.py
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from wordcloud import WordCloud
import networkx as nx
import pandas as pd
from datetime import datetime

from config import PROCESSED_DATA_DIR

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 输出路径
PROFILES_DIR = PROCESSED_DATA_DIR / "academic_profiles"
PROFILES_DIR.mkdir(exist_ok=True)

# 结果文件路径
ACADEMIC_PROFILES_PATH = PROFILES_DIR / "academic_profiles.json"
AUTHOR_NETWORK_PATH = PROFILES_DIR / "author_network.json"
DOMAIN_ANALYSIS_PATH = PROFILES_DIR / "domain_analysis.json"
VISUALIZATIONS_DIR = PROFILES_DIR / "visualizations"
VISUALIZATIONS_DIR.mkdir(exist_ok=True)


class AcademicProfileGenerator:
    """学术画像生成器 - 基于概要数据的统计分析"""

    def __init__(self):
        self.summaries = {}
        self.profiles = {}

    def load_summaries(self):
        """加载概要数据"""
        summaries_path = PROCESSED_DATA_DIR / "document_summaries.json"

        try:
            with open(summaries_path, 'r', encoding='utf-8') as f:
                self.summaries = json.load(f)
            print(f"成功加载 {len(self.summaries)} 个文档概要")
            return True
        except FileNotFoundError:
            print("错误：找不到document_summaries.json文件")
            return False
        except Exception as e:
            print(f"加载概要数据失败: {e}")
            return False

    def clean_field_data(self, field_value):
        """清洗字段数据，移除无效值"""
        if not field_value:
            return []

        if isinstance(field_value, str):
            if field_value.strip() in ["未在原文中明确提及", "", "未知"]:
                return []
            return [field_value.strip()]

        elif isinstance(field_value, list):
            cleaned = []
            for item in field_value:
                if isinstance(item, str) and item.strip():
                    if item.strip() not in ["未在原文中明确提及", "未知"]:
                        cleaned.append(item.strip())
            return cleaned

        return []

    def analyze_authors_and_collaboration(self):
        """分析作者和合作关系"""
        print("分析研究者和合作网络...")

        author_stats = defaultdict(lambda: {
            'paper_count': 0,
            'domains': set(),
            'methods': set(),
            'collaborators': set(),
            'document_types': set(),
            'papers': []
        })

        collaboration_pairs = Counter()

        for doc_id, summary in self.summaries.items():
            doc_type = summary.get('document_type', 'unknown')

            # 获取作者/发明人
            if doc_type == 'patent':
                authors = self.clean_field_data(summary.get('inventors', []))
            else:
                authors = self.clean_field_data(summary.get('authors', []))

            if not authors:
                continue

            # 统计每个作者的信息
            for author in authors:
                author_stats[author]['paper_count'] += 1
                author_stats[author]['document_types'].add(doc_type)
                author_stats[author]['papers'].append(doc_id)

                # 研究领域
                if doc_type == 'patent':
                    domains = self.clean_field_data(summary.get('application_domain', ''))
                else:
                    domains = self.clean_field_data(summary.get('main_topic', ''))

                for domain in domains:
                    author_stats[author]['domains'].add(domain)

                # 方法技术
                if doc_type == 'patent':
                    methods = self.clean_field_data(summary.get('technical_solution', ''))
                else:
                    methods = self.clean_field_data(summary.get('methodology', ''))

                for method in methods:
                    author_stats[author]['methods'].add(method)

            # 记录合作关系
            if len(authors) > 1:
                for i, author1 in enumerate(authors):
                    for author2 in authors[i + 1:]:
                        pair = tuple(sorted([author1, author2]))
                        collaboration_pairs[pair] += 1
                        author_stats[author1]['collaborators'].add(author2)
                        author_stats[author2]['collaborators'].add(author1)

        # 转换为可序列化格式
        author_profiles = {}
        for author, stats in author_stats.items():
            author_profiles[author] = {
                'paper_count': stats['paper_count'],
                'domains': list(stats['domains']),
                'methods': list(stats['methods']),
                'collaborators': list(stats['collaborators']),
                'document_types': list(stats['document_types']),
                'collaboration_strength': len(stats['collaborators']),
                'productivity_level': self.classify_productivity(stats['paper_count']),
                'papers': stats['papers']
            }

        # 合作网络分析
        collaboration_network = {
            'total_authors': len(author_profiles),
            'collaboration_pairs': len(collaboration_pairs),
            'top_collaborations': [
                {'authors': list(pair), 'paper_count': count}
                for pair, count in collaboration_pairs.most_common(10)
            ],
            'most_productive_authors': [
                {'author': author, 'paper_count': profile['paper_count']}
                for author, profile in sorted(author_profiles.items(),
                                              key=lambda x: x[1]['paper_count'], reverse=True)[:10]
            ]
        }

        return author_profiles, collaboration_network

    def classify_productivity(self, paper_count):
        """根据论文数量对研究者进行产出分级"""
        if paper_count >= 5:
            return "高产研究者"
        elif paper_count >= 3:
            return "活跃研究者"
        elif paper_count >= 2:
            return "一般研究者"
        else:
            return "新进研究者"

    def analyze_domains_and_topics(self):
        """分析研究领域和热点"""
        print("分析研究领域和热点...")

        domain_stats = defaultdict(lambda: {
            'paper_count': 0,
            'methods': Counter(),
            'keywords': Counter(),
            'authors': set(),
            'document_types': Counter(),
            'papers': []
        })

        for doc_id, summary in self.summaries.items():
            doc_type = summary.get('document_type', 'unknown')

            # 获取领域信息
            if doc_type == 'patent':
                domains = self.clean_field_data(summary.get('application_domain', ''))
                if not domains:
                    domains = self.clean_field_data(summary.get('application_scenarios', []))
            else:
                domains = self.clean_field_data(summary.get('main_topic', ''))
                if not domains:
                    domains = self.clean_field_data(summary.get('application_domains', []))

            if not domains:
                domains = ['其他领域']

            for domain in domains:
                domain_stats[domain]['paper_count'] += 1
                domain_stats[domain]['document_types'][doc_type] += 1
                domain_stats[domain]['papers'].append(doc_id)

                # 方法统计
                if doc_type == 'patent':
                    methods = self.clean_field_data(summary.get('technical_concepts', []))
                else:
                    methods = self.clean_field_data(summary.get('methodology', ''))
                    methods.extend(self.clean_field_data(summary.get('technical_concepts', [])))

                for method in methods:
                    domain_stats[domain]['methods'][method] += 1

                # 关键词统计
                keywords = self.clean_field_data(summary.get('keywords', []))
                for keyword in keywords:
                    domain_stats[domain]['keywords'][keyword] += 1

                # 作者统计
                if doc_type == 'patent':
                    authors = self.clean_field_data(summary.get('inventors', []))
                else:
                    authors = self.clean_field_data(summary.get('authors', []))

                for author in authors:
                    domain_stats[domain]['authors'].add(author)

        # 转换为可序列化格式
        domain_analysis = {}
        for domain, stats in domain_stats.items():
            domain_analysis[domain] = {
                'paper_count': stats['paper_count'],
                'author_count': len(stats['authors']),
                'top_methods': [
                    {'method': method, 'count': count}
                    for method, count in stats['methods'].most_common(5)
                ],
                'top_keywords': [
                    {'keyword': keyword, 'count': count}
                    for keyword, count in stats['keywords'].most_common(10)
                ],
                'document_types': dict(stats['document_types']),
                'activity_level': self.classify_domain_activity(stats['paper_count']),
                'papers': stats['papers']
            }

        # 全局热点分析
        hotspot_analysis = {
            'total_domains': len(domain_analysis),
            'most_active_domains': [
                {'domain': domain, 'paper_count': analysis['paper_count']}
                for domain, analysis in sorted(domain_analysis.items(),
                                               key=lambda x: x[1]['paper_count'], reverse=True)[:10]
            ]
        }

        return domain_analysis, hotspot_analysis

    def classify_domain_activity(self, paper_count):
        """根据论文数量对研究领域进行活跃度分级"""
        if paper_count >= 10:
            return "热门领域"
        elif paper_count >= 5:
            return "活跃领域"
        elif paper_count >= 2:
            return "新兴领域"
        else:
            return "小众领域"

    def analyze_methods_and_technologies(self):
        """分析方法和技术统计"""
        print("分析技术方法和概念...")

        method_stats = Counter()
        concept_stats = Counter()
        method_domain_map = defaultdict(set)

        for doc_id, summary in self.summaries.items():
            doc_type = summary.get('document_type', 'unknown')

            # 获取方法信息
            if doc_type == 'patent':
                methods = self.clean_field_data(summary.get('technical_solution', ''))
                concepts = self.clean_field_data(summary.get('technical_concepts', []))
                domain = self.clean_field_data(summary.get('application_domain', ''))
            else:
                methods = self.clean_field_data(summary.get('methodology', ''))
                concepts = self.clean_field_data(summary.get('technical_concepts', []))
                domain = self.clean_field_data(summary.get('main_topic', ''))

            # 统计方法
            for method in methods:
                method_stats[method] += 1
                for d in domain:
                    method_domain_map[method].add(d)

            # 统计技术概念
            for concept in concepts:
                concept_stats[concept] += 1

        technology_analysis = {
            'total_methods': len(method_stats),
            'total_concepts': len(concept_stats),
            'top_methods': [
                {
                    'method': method,
                    'count': count,
                    'domains': list(method_domain_map[method])
                }
                for method, count in method_stats.most_common(15)
            ],
            'top_concepts': [
                {'concept': concept, 'count': count}
                for concept, count in concept_stats.most_common(20)
            ]
        }

        return technology_analysis

    def analyze_keywords_and_trends(self):
        """分析关键词和趋势"""
        print("分析关键词和研究趋势...")

        all_keywords = Counter()
        keyword_cooccurrence = defaultdict(Counter)

        for doc_id, summary in self.summaries.items():
            keywords = self.clean_field_data(summary.get('keywords', []))

            # 统计关键词频率
            for keyword in keywords:
                all_keywords[keyword] += 1

            # 统计关键词共现
            for i, kw1 in enumerate(keywords):
                for kw2 in keywords[i + 1:]:
                    keyword_cooccurrence[kw1][kw2] += 1
                    keyword_cooccurrence[kw2][kw1] += 1

        keyword_analysis = {
            'total_unique_keywords': len(all_keywords),
            'top_keywords': [
                {'keyword': kw, 'frequency': freq}
                for kw, freq in all_keywords.most_common(30)
            ],
            'keyword_cooccurrence': {
                kw: [{'keyword': co_kw, 'count': count}
                     for co_kw, count in co_counter.most_common(5)]
                for kw, co_counter in list(keyword_cooccurrence.items())[:10]
            }
        }

        return keyword_analysis

    def generate_document_statistics(self):
        """生成文档统计信息"""
        print("生成文档统计...")

        doc_type_stats = Counter()
        source_type_stats = Counter()

        for doc_id, summary in self.summaries.items():
            doc_type = summary.get('document_type', 'unknown')
            source_type = summary.get('source_type', 'unknown')

            doc_type_stats[doc_type] += 1
            source_type_stats[source_type] += 1

        document_statistics = {
            'total_documents': len(self.summaries),
            'document_types': dict(doc_type_stats),
            'source_types': dict(source_type_stats),
            'processing_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        return document_statistics

    def generate_visualizations(self, profiles_data):
        """生成可视化图表"""
        print("生成可视化图表...")

        try:
            # 1. 研究领域分布饼图
            domain_data = profiles_data['domain_analysis']
            domain_counts = [(domain, data['paper_count'])
                             for domain, data in domain_data.items()]
            domain_counts.sort(key=lambda x: x[1], reverse=True)

            if domain_counts:
                plt.figure(figsize=(12, 8))
                domains, counts = zip(*domain_counts[:10])  # 只显示前10个
                plt.pie(counts, labels=domains, autopct='%1.1f%%', startangle=90)
                plt.title('研究领域分布', fontsize=16, fontweight='bold')
                plt.axis('equal')
                plt.tight_layout()
                plt.savefig(VISUALIZATIONS_DIR / 'domain_distribution.png', dpi=300, bbox_inches='tight')
                plt.close()

            # 2. 技术方法柱状图
            tech_data = profiles_data['technology_analysis']['top_methods'][:10]
            if tech_data:
                plt.figure(figsize=(14, 8))
                methods = [item['method'][:20] + '...' if len(item['method']) > 20
                           else item['method'] for item in tech_data]
                counts = [item['count'] for item in tech_data]

                plt.barh(methods, counts, color='skyblue')
                plt.title('热门技术方法统计', fontsize=16, fontweight='bold')
                plt.xlabel('使用频次')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plt.savefig(VISUALIZATIONS_DIR / 'technology_methods.png', dpi=300, bbox_inches='tight')
                plt.close()

            # 3. 关键词词云图
            keyword_data = profiles_data['keyword_analysis']['top_keywords']
            if keyword_data:
                keyword_freq = {item['keyword']: item['frequency'] for item in keyword_data}

                wordcloud = WordCloud(
                    width=1200, height=600,
                    background_color='white',
                    font_path='simhei.ttf',  # 如果有中文字体文件
                    max_words=100,
                    colormap='viridis'
                ).generate_from_frequencies(keyword_freq)

                plt.figure(figsize=(15, 8))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title('研究关键词词云图', fontsize=16, fontweight='bold', pad=20)
                plt.tight_layout()
                plt.savefig(VISUALIZATIONS_DIR / 'keywords_wordcloud.png', dpi=300, bbox_inches='tight')
                plt.close()

            # 4. 作者产出分布
            author_data = profiles_data['author_profiles']
            productivity_levels = Counter()
            for author, data in author_data.items():
                productivity_levels[data['productivity_level']] += 1

            if productivity_levels:
                plt.figure(figsize=(10, 6))
                levels = list(productivity_levels.keys())
                counts = list(productivity_levels.values())

                plt.bar(levels, counts, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
                plt.title('研究者产出分布', fontsize=16, fontweight='bold')
                plt.ylabel('研究者数量')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(VISUALIZATIONS_DIR / 'author_productivity.png', dpi=300, bbox_inches='tight')
                plt.close()

            print(f"可视化图表已保存到: {VISUALIZATIONS_DIR}")

        except Exception as e:
            print(f"生成可视化图表时出现错误: {e}")
            print("跳过可视化生成，继续处理...")

    def generate_academic_profiles(self):
        """生成完整的学术画像"""
        print("=" * 60)
        print("开始生成学术画像")
        print("=" * 60)

        if not self.load_summaries():
            return

        # 1. 研究者和合作分析
        author_profiles, collaboration_network = self.analyze_authors_and_collaboration()

        # 2. 研究领域分析
        domain_analysis, hotspot_analysis = self.analyze_domains_and_topics()

        # 3. 技术方法分析
        technology_analysis = self.analyze_methods_and_technologies()

        # 4. 关键词趋势分析
        keyword_analysis = self.analyze_keywords_and_trends()

        # 5. 文档统计
        document_statistics = self.generate_document_statistics()

        # 整合所有分析结果
        complete_profiles = {
            'metadata': {
                'generation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'source_documents': len(self.summaries),
                'analysis_version': '1.0'
            },
            'document_statistics': document_statistics,
            'author_profiles': author_profiles,
            'collaboration_network': collaboration_network,
            'domain_analysis': domain_analysis,
            'hotspot_analysis': hotspot_analysis,
            'technology_analysis': technology_analysis,
            'keyword_analysis': keyword_analysis
        }

        # 保存结果
        with open(ACADEMIC_PROFILES_PATH, 'w', encoding='utf-8') as f:
            json.dump(complete_profiles, f, indent=2, ensure_ascii=False)

        # 保存单独的分析文件
        with open(AUTHOR_NETWORK_PATH, 'w', encoding='utf-8') as f:
            json.dump({
                'author_profiles': author_profiles,
                'collaboration_network': collaboration_network
            }, f, indent=2, ensure_ascii=False)

        with open(DOMAIN_ANALYSIS_PATH, 'w', encoding='utf-8') as f:
            json.dump({
                'domain_analysis': domain_analysis,
                'hotspot_analysis': hotspot_analysis
            }, f, indent=2, ensure_ascii=False)

        # 生成可视化图表
        self.generate_visualizations(complete_profiles)

        # 显示统计结果
        self.print_summary(complete_profiles)

        print(f"\n学术画像生成完成！")
        print(f"完整画像保存在: {ACADEMIC_PROFILES_PATH}")
        print(f"作者网络分析: {AUTHOR_NETWORK_PATH}")
        print(f"领域分析: {DOMAIN_ANALYSIS_PATH}")
        print(f"可视化图表: {VISUALIZATIONS_DIR}")

        return complete_profiles

    def print_summary(self, profiles):
        """打印分析摘要"""
        print("\n" + "=" * 60)
        print("学术画像分析摘要")
        print("=" * 60)

        stats = profiles['document_statistics']
        print(f"文档总数: {stats['total_documents']}")
        print(f"文档类型: {stats['document_types']}")

        author_count = len(profiles['author_profiles'])
        print(f"研究者总数: {author_count}")

        domain_count = len(profiles['domain_analysis'])
        print(f"研究领域数: {domain_count}")

        # 显示热门领域
        top_domains = profiles['hotspot_analysis']['most_active_domains'][:5]
        print(f"\n热门研究领域:")
        for i, domain_info in enumerate(top_domains, 1):
            print(f"  {i}. {domain_info['domain']}: {domain_info['paper_count']}篇")

        # 显示高产研究者
        top_authors = profiles['collaboration_network']['most_productive_authors'][:5]
        print(f"\n高产研究者:")
        for i, author_info in enumerate(top_authors, 1):
            print(f"  {i}. {author_info['author']}: {author_info['paper_count']}篇")

        # 显示热门技术
        top_methods = profiles['technology_analysis']['top_methods'][:5]
        print(f"\n热门技术方法:")
        for i, method_info in enumerate(top_methods, 1):
            method_name = method_info['method'][:30] + '...' if len(method_info['method']) > 30 else method_info[
                'method']
            print(f"  {i}. {method_name}: {method_info['count']}次")


def main():
    """主函数"""
    generator = AcademicProfileGenerator()
    profiles = generator.generate_academic_profiles()
    return profiles


if __name__ == "__main__":
    main()