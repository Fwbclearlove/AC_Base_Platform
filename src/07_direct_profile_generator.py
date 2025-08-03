# src/07_direct_profile_generator.py
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime
import networkx as nx
from wordcloud import WordCloud

from config import PROCESSED_DATA_DIR

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 输出路径
DIRECT_PROFILES_DIR = PROCESSED_DATA_DIR / "direct_academic_profiles"
DIRECT_PROFILES_DIR.mkdir(exist_ok=True)

VISUALIZATIONS_DIR = DIRECT_PROFILES_DIR / "visualizations"
VISUALIZATIONS_DIR.mkdir(exist_ok=True)

REPORTS_DIR = DIRECT_PROFILES_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


class DirectProfileGenerator:
    """直接从概要层和结构化层构建学术画像"""

    def __init__(self):
        self.summaries = {}  # 概要层数据
        self.structural_insights = {}  # 结构化层数据
        self.author_name_map = {}  # 姓名标准化映射

    def load_data_sources(self):
        """加载两个数据源"""
        print("加载数据源...")

        # 加载概要层数据
        summaries_path = PROCESSED_DATA_DIR / "document_summaries.json"
        try:
            with open(summaries_path, 'r', encoding='utf-8') as f:
                self.summaries = json.load(f)
            print(f"成功加载概要层数据: {len(self.summaries)} 个文档")
        except FileNotFoundError:
            print("找不到document_summaries.json文件")
            return False
        except Exception as e:
            print(f"加载概要层数据失败: {e}")
            return False

        # 加载结构化数据
        structural_path = PROCESSED_DATA_DIR / "structural_insights" / "structural_insights.json"
        try:
            with open(structural_path, 'r', encoding='utf-8') as f:
                self.structural_insights = json.load(f)
            print(f"成功加载结构化数据: {len(self.structural_insights)} 个文档")
        except FileNotFoundError:
            print("找不到structural_insights.json文件")
            return False
        except Exception as e:
            print(f"加载结构化数据失败: {e}")
            return False

        # 数据覆盖分析
        summary_docs = set(self.summaries.keys())
        structural_docs = set(self.structural_insights.keys())

        overlap = summary_docs & structural_docs
        only_summary = summary_docs - structural_docs
        only_structural = structural_docs - summary_docs

        print(f"数据覆盖分析:")
        print(f"   两种数据都有: {len(overlap)} 个")
        print(f"   仅有概要数据: {len(only_summary)} 个")
        print(f"   仅有结构化数据: {len(only_structural)} 个")
        print(f"   可用文档总数: {len(summary_docs | structural_docs)} 个")

        return True

    def clean_field_data(self, field_value, invalid_values=None):
        """清洗字段数据，移除无效值"""
        if invalid_values is None:
            invalid_values = [
                "原文无此信息", "原文未明确提及", "未在原文中明确提及",
                "未在原文中明确列出", "", "未知", "unknown", "无"
            ]

        if not field_value:
            return []

        if isinstance(field_value, str):
            if field_value.strip() in invalid_values:
                return []
            return [field_value.strip()]

        elif isinstance(field_value, list):
            cleaned = []
            for item in field_value:
                if isinstance(item, str) and item.strip():
                    if item.strip() not in invalid_values:
                        cleaned.append(item.strip())
                elif isinstance(item, dict):
                    # 处理结构化数据中的字典格式
                    item_value = item.get('method_name') or item.get('dataset_name') or item.get('contribution')
                    if item_value and item_value not in invalid_values:
                        cleaned.append(item_value)
            return cleaned

        return []

    def normalize_author_name(self, author_name):
        """标准化作者姓名"""
        if not author_name or author_name.strip() in ["", "未知", "unknown"]:
            return None

        name = author_name.strip()

        # 李睿凡的各种写法统一
        ruifan_variations = [
            "Ruifan Li", "李睿凡", "Li Ruifan", "Li, Ruifan",
            "Ruifan, Li", "ruifan li", "li ruifan"
        ]

        for variation in ruifan_variations:
            if name.lower() == variation.lower():
                return "李睿凡"

        # 其他常见的中英文姓名处理
        # 移除多余空格和标点
        name = name.replace(",", " ").replace(".", " ")
        name = " ".join(name.split())

        # 如果包含中文，优先使用中文版本
        if any('\u4e00' <= char <= '\u9fff' for char in name):
            return name

        # 英文姓名标准化（姓 名 格式）
        parts = name.split()
        if len(parts) == 2:
            # 假设是 "名 姓" 或 "姓 名"，统一为 "姓 名"
            return f"{parts[1]} {parts[0]}" if parts[0][0].isupper() and parts[1][0].isupper() else name

        return name

    def build_author_name_mapping(self):
        """构建作者姓名标准化映射"""
        print("构建作者姓名标准化映射...")

        all_authors = set()

        # 从概要层收集所有作者名
        for doc_id, summary in self.summaries.items():
            doc_type = summary.get('document_type', 'unknown')

            if doc_type == 'patent':
                authors = self.clean_field_data(summary.get('inventors', []))
            else:
                authors = self.clean_field_data(summary.get('authors', []))

            all_authors.update(authors)

        # 从结构化层收集作者名
        for doc_id, structural in self.structural_insights.items():
            metadata = structural.get('document_metadata', {})

            authors = self.clean_field_data(metadata.get('authors', []))
            inventors = self.clean_field_data(metadata.get('inventors', []))
            creators = self.clean_field_data(metadata.get('authors_or_creators', []))

            all_authors.update(authors + inventors + creators)

        # 建立映射
        mapping_stats = defaultdict(list)

        for author in all_authors:
            normalized = self.normalize_author_name(author)
            if normalized:
                self.author_name_map[author] = normalized
                mapping_stats[normalized].append(author)

        # 显示合并情况
        print("姓名合并情况:")
        for normalized_name, original_names in mapping_stats.items():
            if len(original_names) > 1:
                print(f"  {normalized_name}: {', '.join(original_names)}")

        print(f"总共标准化了 {len(all_authors)} 个原始姓名到 {len(set(self.author_name_map.values()))} 个标准姓名")

    def extract_authors_from_document(self, doc_id):
        """从文档中提取并标准化作者信息"""
        authors_info = {
            'standardized_authors': [],
            'institutions': [],
            'raw_authors': [],
            'data_source': 'none'
        }

        # 从概要层获取作者信息
        if doc_id in self.summaries:
            summary = self.summaries[doc_id]
            doc_type = summary.get('document_type', 'unknown')

            if doc_type == 'patent':
                raw_authors = self.clean_field_data(summary.get('inventors', []))
            else:
                raw_authors = self.clean_field_data(summary.get('authors', []))

            if raw_authors:
                authors_info['raw_authors'] = raw_authors
                authors_info['data_source'] = 'summary'

        # 从结构化层补充作者信息
        if doc_id in self.structural_insights:
            structural = self.structural_insights[doc_id]
            metadata = structural.get('document_metadata', {})

            # 如果概要层没有作者，从结构化层获取
            if not authors_info['raw_authors']:
                authors = self.clean_field_data(metadata.get('authors', []))
                inventors = self.clean_field_data(metadata.get('inventors', []))
                creators = self.clean_field_data(metadata.get('authors_or_creators', []))

                raw_authors = authors + inventors + creators
                if raw_authors:
                    authors_info['raw_authors'] = raw_authors
                    authors_info['data_source'] = 'structural'

            # 获取机构信息
            institutions = self.clean_field_data(metadata.get('institutions', []))
            if institutions:
                authors_info['institutions'] = institutions

        # 标准化作者姓名
        for raw_author in authors_info['raw_authors']:
            standardized = self.author_name_map.get(raw_author, raw_author)
            if standardized and standardized not in authors_info['standardized_authors']:
                authors_info['standardized_authors'].append(standardized)

        return authors_info

    def analyze_research_domains(self):
        """分析研究领域"""
        print("分析研究领域...")

        domain_stats = defaultdict(lambda: {
            'paper_count': 0,
            'authors': set(),
            'methods': Counter(),
            'keywords': Counter(),
            'innovations': Counter(),
            'document_types': Counter(),
            'papers': []
        })

        all_docs = set(self.summaries.keys()) | set(self.structural_insights.keys())

        for doc_id in all_docs:
            # 获取领域信息
            domains = []

            # 从概要层获取
            if doc_id in self.summaries:
                summary = self.summaries[doc_id]
                doc_type = summary.get('document_type', 'unknown')

                if doc_type == 'patent':
                    domains.extend(self.clean_field_data(summary.get('application_domain', '')))
                    domains.extend(self.clean_field_data(summary.get('application_scenarios', [])))
                else:
                    domains.extend(self.clean_field_data(summary.get('main_topic', '')))
                    domains.extend(self.clean_field_data(summary.get('application_domains', [])))

            # 如果没有明确领域，使用"其他领域"
            if not domains:
                domains = ['其他领域']

            # 获取作者信息
            authors_info = self.extract_authors_from_document(doc_id)
            standardized_authors = authors_info['standardized_authors']

            for domain in domains:
                domain_stats[domain]['paper_count'] += 1
                domain_stats[domain]['authors'].update(standardized_authors)
                domain_stats[domain]['papers'].append(doc_id)

                # 从概要层获取方法和关键词
                if doc_id in self.summaries:
                    summary = self.summaries[doc_id]
                    doc_type = summary.get('document_type', 'unknown')
                    domain_stats[domain]['document_types'][doc_type] += 1

                    # 方法统计
                    if doc_type == 'patent':
                        methods = self.clean_field_data(summary.get('technical_solution', ''))
                        methods.extend(self.clean_field_data(summary.get('technical_concepts', [])))
                    else:
                        methods = self.clean_field_data(summary.get('methodology', ''))
                        methods.extend(self.clean_field_data(summary.get('technical_concepts', [])))

                    for method in methods:
                        domain_stats[domain]['methods'][method] += 1

                    # 关键词统计
                    keywords = self.clean_field_data(summary.get('keywords', []))
                    for keyword in keywords:
                        domain_stats[domain]['keywords'][keyword] += 1

                # 从结构化层获取创新信息
                if doc_id in self.structural_insights:
                    structural = self.structural_insights[doc_id]
                    innovation_analysis = structural.get('innovation_analysis', {})

                    contributions = self.clean_field_data(innovation_analysis.get('stated_contributions', []))
                    novelty = self.clean_field_data(innovation_analysis.get('stated_novelty', []))

                    for contrib in contributions + novelty:
                        domain_stats[domain]['innovations'][contrib] += 1

        # 转换为可序列化格式
        domain_analysis = {}
        for domain, stats in domain_stats.items():
            domain_analysis[domain] = {
                'paper_count': stats['paper_count'],
                'author_count': len(stats['authors']),
                'unique_authors': list(stats['authors']),
                'top_methods': [
                    {'method': method, 'count': count}
                    for method, count in stats['methods'].most_common(8)
                ],
                'top_keywords': [
                    {'keyword': keyword, 'count': count}
                    for keyword, count in stats['keywords'].most_common(10)
                ],
                'top_innovations': [
                    {'innovation': innovation, 'count': count}
                    for innovation, count in stats['innovations'].most_common(5)
                ],
                'document_types': dict(stats['document_types']),
                'activity_level': self.classify_domain_activity(stats['paper_count']),
                'papers': stats['papers']
            }

        return domain_analysis

    def classify_domain_activity(self, paper_count):
        """领域活跃度分级"""
        if paper_count >= 15:
            return "热门领域"
        elif paper_count >= 8:
            return "活跃领域"
        elif paper_count >= 3:
            return "新兴领域"
        else:
            return "小众领域"

    def analyze_author_profiles(self):
        """分析作者画像"""
        print("分析作者画像...")

        author_profiles = defaultdict(lambda: {
            'basic_info': {
                'total_papers': 0,
                'document_types': Counter(),
                'institutions': set(),
                'raw_names': set(),
                'collaboration_count': 0
            },
            'research_focus': {
                'domains': Counter(),
                'methods': Counter(),
                'keywords': Counter(),
                'concepts': Counter()
            },
            'innovation_analysis': {
                'stated_contributions': [],
                'technical_novelty': [],
                'datasets_used': set(),
                'evaluation_metrics': set(),
                'innovation_strength': 0
            },
            'technical_relationships': {
                'builds_upon': Counter(),
                'compared_with': Counter(),
                'extends': Counter()
            },
            'collaboration_network': {
                'collaborators': set(),
                'institutional_networks': set()
            },
            'paper_list': []
        })

        collaboration_pairs = Counter()
        all_docs = set(self.summaries.keys()) | set(self.structural_insights.keys())

        for doc_id in all_docs:
            # 获取标准化作者信息
            authors_info = self.extract_authors_from_document(doc_id)
            standardized_authors = authors_info['standardized_authors']
            raw_authors = authors_info['raw_authors']
            institutions = authors_info['institutions']

            if not standardized_authors:
                continue

            # 获取文档类型
            doc_type = 'unknown'
            if doc_id in self.summaries:
                doc_type = self.summaries[doc_id].get('document_type', 'unknown')

            for i, author in enumerate(standardized_authors):
                profile = author_profiles[author]

                # 基础信息
                profile['basic_info']['total_papers'] += 1
                profile['basic_info']['document_types'][doc_type] += 1
                profile['basic_info']['institutions'].update(institutions)
                profile['basic_info']['raw_names'].add(raw_authors[i] if i < len(raw_authors) else author)
                profile['paper_list'].append(doc_id)

                # 研究重点（从概要层和结构化层融合）
                self._extract_research_focus_for_author(profile['research_focus'], doc_id, doc_type)

                # 创新分析（主要来自结构化层）
                self._extract_innovation_for_author(profile['innovation_analysis'], doc_id, doc_type)

                # 技术关系（来自结构化层）
                self._extract_technical_relationships_for_author(profile['technical_relationships'], doc_id)

            # 合作关系分析
            if len(standardized_authors) > 1:
                for i, author1 in enumerate(standardized_authors):
                    author_profiles[author1]['basic_info']['collaboration_count'] += len(standardized_authors) - 1

                    for j, author2 in enumerate(standardized_authors):
                        if i != j:
                            author_profiles[author1]['collaboration_network']['collaborators'].add(author2)

                            # 记录合作对
                            if i < j:
                                pair = tuple(sorted([author1, author2]))
                                collaboration_pairs[pair] += 1

                # 机构网络
                for author in standardized_authors:
                    author_profiles[author]['collaboration_network']['institutional_networks'].update(institutions)

        # 处理并生成最终画像
        processed_profiles = self._process_author_profiles_final(author_profiles, collaboration_pairs)

        return processed_profiles

    def _extract_research_focus_for_author(self, focus_dict, doc_id, doc_type):
        """为作者提取研究重点"""
        # 从概要层获取
        if doc_id in self.summaries:
            summary = self.summaries[doc_id]

            if doc_type == 'patent':
                domains = self.clean_field_data(summary.get('application_domain', ''))
                domains.extend(self.clean_field_data(summary.get('application_scenarios', [])))
                methods = self.clean_field_data(summary.get('technical_solution', ''))
                methods.extend(self.clean_field_data(summary.get('technical_concepts', [])))
            else:
                domains = self.clean_field_data(summary.get('main_topic', ''))
                domains.extend(self.clean_field_data(summary.get('application_domains', [])))
                methods = self.clean_field_data(summary.get('methodology', ''))
                methods.extend(self.clean_field_data(summary.get('technical_concepts', [])))

            keywords = self.clean_field_data(summary.get('keywords', []))

            # 更新计数
            for domain in domains:
                focus_dict['domains'][domain] += 1
            for method in methods:
                focus_dict['methods'][method] += 1
            for keyword in keywords:
                focus_dict['keywords'][keyword] += 1

        # 从结构化层补充
        if doc_id in self.structural_insights:
            structural = self.structural_insights[doc_id]

            # 获取技术概念
            if doc_type == 'experimental_paper':
                exp_setup = structural.get('experimental_setup', {})
                datasets = exp_setup.get('datasets_used', [])
                for dataset in datasets:
                    if isinstance(dataset, dict):
                        name = dataset.get('dataset_name', '')
                        if name and name not in ["原文无此信息", "原文未明确提及"]:
                            focus_dict['concepts'][f"数据集:{name}"] += 1

    def _extract_innovation_for_author(self, innovation_dict, doc_id, doc_type):
        """为作者提取创新信息"""
        if doc_id not in self.structural_insights:
            return

        structural = self.structural_insights[doc_id]

        # 提取贡献声明
        innovation_analysis = structural.get('innovation_analysis', {})
        contributions = self.clean_field_data(innovation_analysis.get('stated_contributions', []))
        novelty = self.clean_field_data(innovation_analysis.get('stated_novelty', []))

        innovation_dict['stated_contributions'].extend(contributions)
        innovation_dict['technical_novelty'].extend(novelty)

        # 计算创新强度
        innovation_dict['innovation_strength'] += len(contributions) + len(novelty)

        # 实验信息（如果是实验性论文）
        if doc_type == 'experimental_paper':
            exp_setup = structural.get('experimental_setup', {})

            datasets = exp_setup.get('datasets_used', [])
            for dataset in datasets:
                if isinstance(dataset, dict):
                    name = dataset.get('dataset_name', '')
                    if name and name not in ["原文无此信息", "原文未明确提及"]:
                        innovation_dict['datasets_used'].add(name)

            metrics = exp_setup.get('evaluation_metrics', [])
            for metric in self.clean_field_data(metrics):
                innovation_dict['evaluation_metrics'].add(metric)

    def _extract_technical_relationships_for_author(self, tech_rel_dict, doc_id):
        """为作者提取技术关系"""
        if doc_id not in self.structural_insights:
            return

        structural = self.structural_insights[doc_id]
        tech_relationships = structural.get('technical_relationships', {})

        # 基于的方法
        base_methods = tech_relationships.get('base_methods', [])
        for method_info in base_methods:
            method_name = None
            if isinstance(method_info, dict):
                method_name = method_info.get('method_name', '')
            elif isinstance(method_info, str):
                method_name = method_info

            if method_name and method_name not in ["原文无此信息", "原文未明确提及"]:
                tech_rel_dict['builds_upon'][method_name] += 1

        # 对比的方法
        compared_methods = tech_relationships.get('compared_methods', [])
        for method_info in compared_methods:
            method_name = None
            if isinstance(method_info, dict):
                method_name = method_info.get('method_name', '')
            elif isinstance(method_info, str):
                method_name = method_info

            if method_name and method_name not in ["原文无此信息", "原文未明确提及"]:
                tech_rel_dict['compared_with'][method_name] += 1

    def _process_author_profiles_final(self, author_profiles, collaboration_pairs):
        """处理作者画像为最终格式"""
        processed_profiles = {}

        for author, profile in author_profiles.items():
            processed_profile = {
                'basic_info': {
                    'standardized_name': author,
                    'raw_names': list(profile['basic_info']['raw_names']),
                    'total_papers': profile['basic_info']['total_papers'],
                    'document_types': dict(profile['basic_info']['document_types']),
                    'institutions': list(profile['basic_info']['institutions']),
                    'collaboration_count': profile['basic_info']['collaboration_count'],
                    'productivity_level': self._classify_productivity(profile['basic_info']['total_papers'])
                },
                'research_focus': {
                    'primary_domains': [{'domain': k, 'count': v} for k, v in
                                        profile['research_focus']['domains'].most_common(5)],
                    'core_methods': [{'method': k, 'count': v} for k, v in
                                     profile['research_focus']['methods'].most_common(8)],
                    'frequent_keywords': [{'keyword': k, 'count': v} for k, v in
                                          profile['research_focus']['keywords'].most_common(10)],
                    'technical_concepts': [{'concept': k, 'count': v} for k, v in
                                           profile['research_focus']['concepts'].most_common(8)]
                },
                'innovation_profile': {
                    'total_contributions': len(profile['innovation_analysis']['stated_contributions']),
                    'total_novelty_claims': len(profile['innovation_analysis']['technical_novelty']),
                    'innovation_strength': profile['innovation_analysis']['innovation_strength'],
                    'datasets_expertise': list(profile['innovation_analysis']['datasets_used']),
                    'evaluation_metrics_used': list(profile['innovation_analysis']['evaluation_metrics']),
                    'innovation_level': self._classify_innovation_level(
                        profile['innovation_analysis']['innovation_strength'])
                },
                'technical_network': {
                    'builds_upon_methods': [{'method': k, 'count': v} for k, v in
                                            profile['technical_relationships']['builds_upon'].most_common(5)],
                    'compared_methods': [{'method': k, 'count': v} for k, v in
                                         profile['technical_relationships']['compared_with'].most_common(5)],
                    'extends_theories': [{'theory': k, 'count': v} for k, v in
                                         profile['technical_relationships']['extends'].most_common(3)]
                },
                'collaboration_network': {
                    'direct_collaborators': list(profile['collaboration_network']['collaborators']),
                    'institutional_networks': list(profile['collaboration_network']['institutional_networks']),
                    'collaboration_strength': len(profile['collaboration_network']['collaborators']),
                    'network_level': self._classify_collaboration_level(
                        len(profile['collaboration_network']['collaborators']))
                },
                'paper_list': profile['paper_list']
            }

            processed_profiles[author] = processed_profile

        return processed_profiles

    def _classify_productivity(self, paper_count):
        """产出水平分级"""
        if paper_count >= 10:
            return "高产研究者"
        elif paper_count >= 6:
            return "活跃研究者"
        elif paper_count >= 3:
            return "稳定研究者"
        else:
            return "新兴研究者"

    def _classify_innovation_level(self, innovation_strength):
        """创新水平分级"""
        if innovation_strength >= 15:
            return "创新引领者"
        elif innovation_strength >= 8:
            return "创新活跃者"
        elif innovation_strength >= 3:
            return "创新参与者"
        else:
            return "创新学习者"

    def _classify_collaboration_level(self, collaborator_count):
        """合作水平分级"""
        if collaborator_count >= 15:
            return "合作网络核心"
        elif collaborator_count >= 8:
            return "合作活跃者"
        elif collaborator_count >= 3:
            return "团队合作者"
        else:
            return "独立研究者"

    def analyze_technology_innovation(self):
        """分析技术创新"""
        print("分析技术创新...")

        innovation_stats = {
            'contribution_types': Counter(),
            'novelty_areas': Counter(),
            'technical_methods': Counter(),
            'evaluation_metrics': Counter(),
            'datasets': Counter(),
            'method_evolution': defaultdict(list)
        }

        for doc_id in self.structural_insights:
            structural = self.structural_insights[doc_id]

            # 创新分析
            innovation_analysis = structural.get('innovation_analysis', {})
            contributions = self.clean_field_data(innovation_analysis.get('stated_contributions', []))
            novelty = self.clean_field_data(innovation_analysis.get('stated_novelty', []))

            for contrib in contributions:
                innovation_stats['contribution_types'][contrib] += 1

            for nov in novelty:
                innovation_stats['novelty_areas'][nov] += 1

            # 技术方法统计
            tech_relationships = structural.get('technical_relationships', {})
            base_methods = tech_relationships.get('base_methods', [])
            for method_info in base_methods:
                method_name = None
                if isinstance(method_info, dict):
                    method_name = method_info.get('method_name', '')
                elif isinstance(method_info, str):
                    method_name = method_info

                if method_name and method_name not in ["原文无此信息", "原文未明确提及"]:
                    innovation_stats['technical_methods'][method_name] += 1

            # 实验生态系统
            exp_setup = structural.get('experimental_setup', {})
            if exp_setup:
                datasets = exp_setup.get('datasets_used', [])
                for dataset in datasets:
                    if isinstance(dataset, dict):
                        name = dataset.get('dataset_name', '')
                        if name and name not in ["原文无此信息", "原文未明确提及"]:
                            innovation_stats['datasets'][name] += 1

                metrics = exp_setup.get('evaluation_metrics', [])
                for metric in self.clean_field_data(metrics):
                    innovation_stats['evaluation_metrics'][metric] += 1

        # 转换为分析结果
        technology_analysis = {
            'innovation_landscape': {
                'top_contribution_types': [
                    {'contribution': k, 'frequency': v}
                    for k, v in innovation_stats['contribution_types'].most_common(20)
                ],
                'emerging_novelty_areas': [
                    {'area': k, 'frequency': v}
                    for k, v in innovation_stats['novelty_areas'].most_common(15)
                ]
            },
            'technical_methods': {
                'popular_base_methods': [
                    {'method': k, 'usage_count': v}
                    for k, v in innovation_stats['technical_methods'].most_common(15)
                ]
            },
            'experimental_ecosystem': {
                'popular_datasets': [
                    {'dataset': k, 'usage_count': v}
                    for k, v in innovation_stats['datasets'].most_common(12)
                ],
                'common_metrics': [
                    {'metric': k, 'usage_count': v}
                    for k, v in innovation_stats['evaluation_metrics'].most_common(12)
                ]
            }
        }

        return technology_analysis

    def generate_comprehensive_statistics(self):
        """生成综合统计"""
        summary_count = len(self.summaries)
        structural_count = len(self.structural_insights)
        total_unique = len(set(self.summaries.keys()) | set(self.structural_insights.keys()))

        # 文档类型统计
        doc_types = Counter()
        for summary in self.summaries.values():
            doc_types[summary.get('document_type', 'unknown')] += 1

        statistics = {
            'data_coverage': {
                'summary_documents': summary_count,
                'structural_documents': structural_count,
                'total_unique_documents': total_unique,
                'overlap_documents': len(set(self.summaries.keys()) & set(self.structural_insights.keys()))
            },
            'document_types': dict(doc_types),
            'author_name_mapping': {
                'original_names': len(self.author_name_map),
                'standardized_names': len(set(self.author_name_map.values())),
                'merge_ratio': len(self.author_name_map) / len(
                    set(self.author_name_map.values())) if self.author_name_map else 0
            },
            'generation_metadata': {
                'generation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'data_sources': ['document_summaries.json', 'structural_insights.json'],
                'processing_method': 'direct_fusion'
            }
        }

        return statistics

    def create_visualizations(self, profiles_data):
        """创建可视化图表"""
        print("创建可视化图表...")

        try:
            # 1. 研究领域分布
            domain_data = profiles_data['domain_analysis']
            if domain_data:
                domain_counts = [(domain, data['paper_count']) for domain, data in domain_data.items()]
                domain_counts.sort(key=lambda x: x[1], reverse=True)

                if domain_counts:
                    plt.figure(figsize=(14, 10))
                    domains, counts = zip(*domain_counts[:12])

                    # 创建横向柱状图
                    colors = plt.cm.Set3(np.linspace(0, 1, len(domains)))
                    bars = plt.barh(domains, counts, color=colors)

                    plt.title('研究领域分布', fontsize=16, fontweight='bold')
                    plt.xlabel('论文数量')

                    # 添加数值标签
                    for bar, count in zip(bars, counts):
                        plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                                 str(count), ha='left', va='center')

                    plt.tight_layout()
                    plt.savefig(VISUALIZATIONS_DIR / 'domain_distribution.png', dpi=300, bbox_inches='tight')
                    plt.close()

            # 2. 作者生产力分布
            author_profiles = profiles_data['author_profiles']
            if author_profiles:
                productivity_levels = Counter()
                paper_counts = []

                for author, profile in author_profiles.items():
                    level = profile['basic_info']['productivity_level']
                    papers = profile['basic_info']['total_papers']
                    productivity_levels[level] += 1
                    paper_counts.append(papers)

                # 生产力等级饼图
                plt.figure(figsize=(10, 8))
                levels = list(productivity_levels.keys())
                counts = list(productivity_levels.values())
                colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

                plt.pie(counts, labels=levels, autopct='%1.1f%%', colors=colors, startangle=90)
                plt.title('研究者生产力分布', fontsize=16, fontweight='bold')
                plt.axis('equal')
                plt.tight_layout()
                plt.savefig(VISUALIZATIONS_DIR / 'productivity_distribution.png', dpi=300, bbox_inches='tight')
                plt.close()

            # 3. 技术创新热力图
            tech_data = profiles_data['technology_analysis']['innovation_landscape']['top_contribution_types'][:15]
            if tech_data:
                plt.figure(figsize=(16, 10))
                contributions = [item['contribution'][:60] + '...' if len(item['contribution']) > 60
                                 else item['contribution'] for item in tech_data]
                frequencies = [item['frequency'] for item in tech_data]

                plt.barh(contributions, frequencies, color='lightcoral')
                plt.title('热门技术贡献类型', fontsize=16, fontweight='bold')
                plt.xlabel('出现频次')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plt.savefig(VISUALIZATIONS_DIR / 'innovation_contributions.png', dpi=300, bbox_inches='tight')
                plt.close()

            # 4. 合作网络强度
            collaboration_levels = Counter()
            for author, profile in author_profiles.items():
                level = profile['collaboration_network']['network_level']
                collaboration_levels[level] += 1

            if collaboration_levels:
                plt.figure(figsize=(12, 6))
                levels = list(collaboration_levels.keys())
                counts = list(collaboration_levels.values())
                colors = ['#ffeaa7', '#fab1a0', '#fd79a8', '#e17055']

                plt.bar(levels, counts, color=colors)
                plt.title('研究者合作网络强度分布', fontsize=16, fontweight='bold')
                plt.ylabel('研究者数量')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(VISUALIZATIONS_DIR / 'collaboration_levels.png', dpi=300, bbox_inches='tight')
                plt.close()

            print(f"可视化图表已保存到: {VISUALIZATIONS_DIR}")

        except Exception as e:
            print(f"生成可视化时出现错误: {e}")

    def create_summary_report(self, profiles_data):
        """创建总结报告"""
        print("生成总结报告...")

        stats = profiles_data['statistics']
        author_count = len(profiles_data['author_profiles'])
        domain_count = len(profiles_data['domain_analysis'])

        # Markdown报告
        md_content = f"""# 学术团队画像分析报告（直接构建版）

## 数据概览

- **分析时间**: {stats['generation_metadata']['generation_time']}
- **数据来源**: 概要层 + 结构化层直接融合
- **文档总数**: {stats['data_coverage']['total_unique_documents']}
- **重叠文档**: {stats['data_coverage']['overlap_documents']}

## 研究者分析

- **研究者总数**: {author_count}
- **姓名标准化**: {stats['author_name_mapping']['original_names']} → {stats['author_name_mapping']['standardized_names']}
- **合并比例**: {stats['author_name_mapping']['merge_ratio']:.2f}

## 研究领域分析

- **活跃领域数**: {domain_count}

### 热门研究领域TOP5
"""

        # 添加热门领域
        domain_analysis = profiles_data['domain_analysis']
        top_domains = sorted(domain_analysis.items(), key=lambda x: x[1]['paper_count'], reverse=True)[:5]

        for i, (domain, data) in enumerate(top_domains, 1):
            md_content += f"{i}. **{domain}**: {data['paper_count']}篇论文, {data['author_count']}位研究者\n"

        md_content += "\n### 高产研究者TOP5\n"

        # 添加高产研究者
        author_profiles = profiles_data['author_profiles']
        top_authors = sorted(author_profiles.items(), key=lambda x: x[1]['basic_info']['total_papers'], reverse=True)[
                      :5]

        for i, (author, profile) in enumerate(top_authors, 1):
            papers = profile['basic_info']['total_papers']
            level = profile['basic_info']['productivity_level']
            md_content += f"{i}. **{author}**: {papers}篇论文 ({level})\n"

        md_content += "\n### 热门技术创新TOP5\n"

        # 添加技术创新
        innovation_data = profiles_data['technology_analysis']['innovation_landscape']['top_contribution_types'][:5]
        for i, contrib in enumerate(innovation_data, 1):
            md_content += f"{i}. {contrib['contribution'][:80]}... (频次: {contrib['frequency']})\n"

        md_content += f"""

## 技术分析

### 实验生态系统
- **热门数据集**: {len(profiles_data['technology_analysis']['experimental_ecosystem']['popular_datasets'])}个
- **常用指标**: {len(profiles_data['technology_analysis']['experimental_ecosystem']['common_metrics'])}个

## 数据质量

- **原始姓名数**: {stats['author_name_mapping']['original_names']}
- **标准化后**: {stats['author_name_mapping']['standardized_names']}
- **重复率**: {(1 - 1 / stats['author_name_mapping']['merge_ratio']) * 100:.1f}%

---

*本报告基于概要层和结构化层数据直接构建，确保数据准确性和完整性*
"""

        # 保存报告
        with open(REPORTS_DIR / 'direct_profile_summary.md', 'w', encoding='utf-8') as f:
            f.write(md_content)

        print(f"总结报告已保存到: {REPORTS_DIR / 'direct_profile_summary.md'}")

    def generate_direct_profiles(self):
        """生成直接学术画像"""
        print("=" * 80)
        print("直接学术画像生成器")
        print("基于概要层和结构化层数据直接构建")
        print("=" * 80)

        # 1. 加载数据源
        if not self.load_data_sources():
            return None

        # 2. 构建姓名标准化映射
        self.build_author_name_mapping()

        # 3. 分析研究领域
        print("\n" + "=" * 50)
        domain_analysis = self.analyze_research_domains()

        # 4. 分析作者画像
        print("\n" + "=" * 50)
        author_profiles = self.analyze_author_profiles()

        # 5. 分析技术创新
        print("\n" + "=" * 50)
        technology_analysis = self.analyze_technology_innovation()

        # 6. 生成综合统计
        statistics = self.generate_comprehensive_statistics()

        # 整合所有分析结果
        complete_profiles = {
            'metadata': {
                'generation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'version': 'direct_build_v1.0',
                'data_sources': ['概要层', '结构化层'],
                'method': '直接融合构建'
            },
            'statistics': statistics,
            'author_profiles': author_profiles,
            'domain_analysis': domain_analysis,
            'technology_analysis': technology_analysis,
            'author_name_mapping': self.author_name_map
        }

        # 保存完整画像
        profiles_path = DIRECT_PROFILES_DIR / "direct_academic_profiles.json"
        with open(profiles_path, 'w', encoding='utf-8') as f:
            json.dump(complete_profiles, f, indent=2, ensure_ascii=False)

        # 保存分别的分析文件
        with open(DIRECT_PROFILES_DIR / "author_profiles.json", 'w', encoding='utf-8') as f:
            json.dump(author_profiles, f, indent=2, ensure_ascii=False)

        with open(DIRECT_PROFILES_DIR / "domain_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(domain_analysis, f, indent=2, ensure_ascii=False)

        with open(DIRECT_PROFILES_DIR / "technology_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(technology_analysis, f, indent=2, ensure_ascii=False)

        # 创建可视化
        print("\n" + "=" * 50)
        self.create_visualizations(complete_profiles)

        # 生成报告
        print("\n" + "=" * 50)
        self.create_summary_report(complete_profiles)

        # 显示统计摘要
        self.print_final_summary(complete_profiles)

        print(f"\n" + "=" * 80)
        print("直接学术画像生成完成！")
        print(f"完整画像: {profiles_path}")
        print(f"作者画像: {DIRECT_PROFILES_DIR / 'author_profiles.json'}")
        print(f"领域分析: {DIRECT_PROFILES_DIR / 'domain_analysis.json'}")
        print(f"技术分析: {DIRECT_PROFILES_DIR / 'technology_analysis.json'}")
        print(f"可视化: {VISUALIZATIONS_DIR}")
        print(f"报告: {REPORTS_DIR}")
        print("=" * 80)

        return complete_profiles

    def print_final_summary(self, profiles):
        """打印最终摘要"""
        print("\n" + "=" * 60)
        print("直接构建学术画像摘要")
        print("=" * 60)

        stats = profiles['statistics']
        print(f"数据覆盖:")
        print(f"   概要文档: {stats['data_coverage']['summary_documents']}")
        print(f"   结构化文档: {stats['data_coverage']['structural_documents']}")
        print(f"   总计文档: {stats['data_coverage']['total_unique_documents']}")
        print(f"   重叠文档: {stats['data_coverage']['overlap_documents']}")

        print(f"\n姓名标准化效果:")
        print(f"   原始姓名: {stats['author_name_mapping']['original_names']}")
        print(f"   标准化后: {stats['author_name_mapping']['standardized_names']}")
        print(f"   合并效率: {stats['author_name_mapping']['merge_ratio']:.2f}")

        author_count = len(profiles['author_profiles'])
        domain_count = len(profiles['domain_analysis'])
        print(f"\n分析结果:")
        print(f"   研究者: {author_count} 位")
        print(f"   研究领域: {domain_count} 个")

        # 显示顶级研究者
        author_profiles = profiles['author_profiles']
        top_productive = sorted(author_profiles.items(),
                                key=lambda x: x[1]['basic_info']['total_papers'],
                                reverse=True)[:5]
        print(f"\n高产研究者TOP5:")
        for i, (author, profile) in enumerate(top_productive, 1):
            papers = profile['basic_info']['total_papers']
            raw_names = profile['basic_info']['raw_names']
            level = profile['basic_info']['productivity_level']
            name_info = f" (合并: {', '.join(raw_names)})" if len(raw_names) > 1 else ""
            print(f"   {i}. {author}: {papers}篇 ({level}){name_info}")


def main():
    """主函数"""
    generator = DirectProfileGenerator()
    profiles = generator.generate_direct_profiles()
    return profiles


if __name__ == "__main__":
    main()