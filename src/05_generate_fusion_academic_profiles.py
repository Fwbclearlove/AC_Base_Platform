# src/05_generate_fusion_academic_profiles.py
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
PROFILES_DIR = PROCESSED_DATA_DIR / "fusion_academic_profiles"
PROFILES_DIR.mkdir(exist_ok=True)

# 结果文件路径
FUSION_PROFILES_PATH = PROFILES_DIR / "fusion_academic_profiles.json"
DEEP_AUTHOR_ANALYSIS_PATH = PROFILES_DIR / "deep_author_analysis.json"
INNOVATION_LANDSCAPE_PATH = PROFILES_DIR / "innovation_landscape.json"
TECHNICAL_NETWORK_PATH = PROFILES_DIR / "technical_network.json"
VISUALIZATIONS_DIR = PROFILES_DIR / "visualizations"
VISUALIZATIONS_DIR.mkdir(exist_ok=True)


class FusionAcademicProfileGenerator:
    """融合学术画像生成器 - 整合概要层和结构化层信息"""

    def __init__(self):
        self.summaries = {}  # 概要层信息
        self.structural_insights = {}  # 结构化信息
        self.fusion_profiles = {}

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

        # 分析数据覆盖情况
        summary_docs = set(self.summaries.keys())
        structural_docs = set(self.structural_insights.keys())

        overlap = summary_docs & structural_docs
        only_summary = summary_docs - structural_docs
        only_structural = structural_docs - summary_docs

        print(f"数据覆盖分析:")
        print(f"   两种数据都有: {len(overlap)} 个文档")
        print(f"   仅有概要数据: {len(only_summary)} 个文档")
        print(f"   仅有结构化数据: {len(only_structural)} 个文档")
        print(f"   总计可用文档: {len(summary_docs | structural_docs)} 个")

        return True

    def clean_field_data(self, field_value, invalid_values=None):
        """清洗字段数据，移除无效值"""
        if invalid_values is None:
            invalid_values = ["原文无此信息", "原文未明确提及", "未在原文中明确提及", "", "未知"]

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
                    cleaned.append(item)
            return cleaned

        return []

    def extract_authors_from_document(self, doc_id):
        """从两个数据源中提取作者信息"""
        authors_info = {
            'authors': [],
            'institutions': [],
            'source': 'none'
        }

        # 优先从概要层获取基本作者信息
        if doc_id in self.summaries:
            summary = self.summaries[doc_id]
            doc_type = summary.get('document_type', 'unknown')

            if doc_type == 'patent':
                authors = self.clean_field_data(summary.get('inventors', []))
            else:
                authors = self.clean_field_data(summary.get('authors', []))

            if authors:
                authors_info['authors'] = authors
                authors_info['source'] = 'summary'

        # 从结构化数据中补充机构信息
        if doc_id in self.structural_insights:
            structural = self.structural_insights[doc_id]
            metadata = structural.get('document_metadata', {})

            # 如果概要层没有作者信息，从结构化数据获取
            if not authors_info['authors']:
                if 'authors' in metadata:
                    authors = self.clean_field_data(metadata.get('authors', []))
                elif 'inventors' in metadata:
                    authors = self.clean_field_data(metadata.get('inventors', []))
                else:
                    authors = self.clean_field_data(metadata.get('authors_or_creators', []))

                if authors:
                    authors_info['authors'] = authors
                    authors_info['source'] = 'structural'

            # 获取机构信息
            institutions = self.clean_field_data(metadata.get('institutions', []))
            if institutions:
                authors_info['institutions'] = institutions

        return authors_info

    def analyze_deep_author_profiles(self):
        """深度分析研究者画像"""
        print(" 深度分析研究者画像...")

        author_profiles = defaultdict(lambda: {
            'basic_info': {
                'total_papers': 0,
                'document_types': Counter(),
                'institutions': set(),
                'collaboration_count': 0
            },
            'research_focus': {
                'domains': Counter(),
                'methods': Counter(),
                'concepts': Counter(),
                'keywords': Counter()
            },
            'innovation_analysis': {
                'stated_contributions': [],
                'technical_novelty': [],
                'limitations_acknowledged': [],
                'datasets_used': set(),
                'evaluation_metrics': set()
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

        # 遍历所有文档
        all_docs = set(self.summaries.keys()) | set(self.structural_insights.keys())

        for doc_id in all_docs:
            # 获取作者信息
            authors_info = self.extract_authors_from_document(doc_id)
            authors = authors_info['authors']

            if not authors:
                continue

            # 从概要层获取基础信息
            summary = self.summaries.get(doc_id, {})
            doc_type = summary.get('document_type', 'unknown')

            # 从结构化层获取详细信息
            structural = self.structural_insights.get(doc_id, {})

            for author in authors:
                profile = author_profiles[author]

                # 基础信息
                profile['basic_info']['total_papers'] += 1
                profile['basic_info']['document_types'][doc_type] += 1
                profile['basic_info']['institutions'].update(authors_info['institutions'])
                profile['paper_list'].append(doc_id)

                # 研究重点（融合两个数据源）
                self._extract_research_focus(profile['research_focus'], summary, structural, doc_type)

                # 创新分析（主要来自结构化数据）
                self._extract_innovation_analysis(profile['innovation_analysis'], structural, doc_type)

                # 技术关系（主要来自结构化数据）
                self._extract_technical_relationships(profile['technical_relationships'], structural)

            # 合作关系分析
            if len(authors) > 1:
                for i, author1 in enumerate(authors):
                    author_profiles[author1]['basic_info']['collaboration_count'] += len(authors) - 1
                    for author2 in authors[i + 1:]:
                        pair = tuple(sorted([author1, author2]))
                        collaboration_pairs[pair] += 1
                        author_profiles[author1]['collaboration_network']['collaborators'].add(author2)
                        author_profiles[author2]['collaboration_network']['collaborators'].add(author1)

                # 机构网络
                institutions = authors_info['institutions']
                for author in authors:
                    author_profiles[author]['collaboration_network']['institutional_networks'].update(institutions)

        # 转换为可序列化格式并计算衍生指标
        serializable_profiles = self._process_author_profiles(author_profiles, collaboration_pairs)

        return serializable_profiles

    def _extract_research_focus(self, focus_dict, summary, structural, doc_type):
        """提取研究重点信息"""
        # 从概要层获取领域和方法
        if doc_type == 'patent':
            domains = self.clean_field_data(summary.get('application_domain', ''))
            methods = self.clean_field_data(summary.get('technical_solution', ''))
        else:
            domains = self.clean_field_data(summary.get('main_topic', ''))
            methods = self.clean_field_data(summary.get('methodology', ''))

        concepts = self.clean_field_data(summary.get('technical_concepts', []))
        keywords = self.clean_field_data(summary.get('keywords', []))

        # 更新计数
        for domain in domains:
            focus_dict['domains'][domain] += 1
        for method in methods:
            focus_dict['methods'][method] += 1
        for concept in concepts:
            focus_dict['concepts'][concept] += 1
        for keyword in keywords:
            focus_dict['keywords'][keyword] += 1

        # 从结构化数据补充
        if structural:
            # 从不同文档类型的结构化数据中提取
            if doc_type == 'experimental_paper':
                exp_setup = structural.get('experimental_setup', {})
                datasets = exp_setup.get('datasets_used', [])
                for dataset in datasets:
                    if isinstance(dataset, dict):
                        name = dataset.get('dataset_name', '')
                        if name and name != '原文无此信息':
                            focus_dict['concepts'][f"数据集:{name}"] += 1

    def _extract_innovation_analysis(self, innovation_dict, structural, doc_type):
        """提取创新分析信息"""
        if not structural:
            return

        # 提取贡献声明
        innovation_analysis = structural.get('innovation_analysis', {})
        contributions = self.clean_field_data(innovation_analysis.get('stated_contributions', []))
        novelty = self.clean_field_data(innovation_analysis.get('stated_novelty', []))

        innovation_dict['stated_contributions'].extend(contributions)
        innovation_dict['technical_novelty'].extend(novelty)

        # 理论性论文的特殊处理
        if doc_type == 'theoretical_paper':
            theoretical_contrib = structural.get('theoretical_contributions', {})
            math_contrib = self.clean_field_data(theoretical_contrib.get('mathematical_contributions', []))
            innovation_dict['technical_novelty'].extend(math_contrib)

        # 局限性
        limitations = structural.get('limitations', {})
        if limitations:
            ack_limitations = self.clean_field_data(limitations.get('acknowledged_limitations', []))
            innovation_dict['limitations_acknowledged'].extend(ack_limitations)

        # 实验信息
        if doc_type == 'experimental_paper':
            exp_setup = structural.get('experimental_setup', {})
            datasets = exp_setup.get('datasets_used', [])
            metrics = exp_setup.get('evaluation_metrics', [])

            for dataset in datasets:
                if isinstance(dataset, dict):
                    name = dataset.get('dataset_name', '')
                    if name and name != '原文无此信息':
                        innovation_dict['datasets_used'].add(name)

            for metric in self.clean_field_data(metrics):
                innovation_dict['evaluation_metrics'].add(metric)

    def _extract_technical_relationships(self, tech_rel_dict, structural):
        """提取技术关系信息"""
        if not structural:
            return

        tech_relationships = structural.get('technical_relationships', {})

        # 基于的方法
        base_methods = tech_relationships.get('base_methods', [])
        for method_info in base_methods:
            if isinstance(method_info, dict):
                method_name = method_info.get('method_name', '')
                if method_name and method_name != '原文无此信息':
                    tech_rel_dict['builds_upon'][method_name] += 1
            elif isinstance(method_info, str) and method_info != '原文无此信息':
                tech_rel_dict['builds_upon'][method_info] += 1

        # 对比的方法
        compared_methods = tech_relationships.get('compared_methods', [])
        for method_info in compared_methods:
            if isinstance(method_info, dict):
                method_name = method_info.get('method_name', '')
                if method_name and method_name != '原文无此信息':
                    tech_rel_dict['compared_with'][method_name] += 1

        # 扩展的理论（主要来自理论性论文）
        extends = tech_relationships.get('extends', [])
        for theory in self.clean_field_data(extends):
            tech_rel_dict['extends'][theory] += 1

    def _process_author_profiles(self, author_profiles, collaboration_pairs):
        """处理作者画像，转换为可序列化格式"""
        processed_profiles = {}

        for author, profile in author_profiles.items():
            processed_profile = {
                'basic_info': {
                    'total_papers': profile['basic_info']['total_papers'],
                    'document_types': dict(profile['basic_info']['document_types']),
                    'institutions': list(profile['basic_info']['institutions']),
                    'collaboration_count': profile['basic_info']['collaboration_count'],
                    'productivity_level': self._classify_productivity(profile['basic_info']['total_papers'])
                },
                'research_focus': {
                    'top_domains': [{'domain': k, 'count': v} for k, v in
                                    profile['research_focus']['domains'].most_common(5)],
                    'top_methods': [{'method': k, 'count': v} for k, v in
                                    profile['research_focus']['methods'].most_common(5)],
                    'top_concepts': [{'concept': k, 'count': v} for k, v in
                                     profile['research_focus']['concepts'].most_common(10)],
                    'top_keywords': [{'keyword': k, 'count': v} for k, v in
                                     profile['research_focus']['keywords'].most_common(10)]
                },
                'innovation_profile': {
                    'total_contributions': len(profile['innovation_analysis']['stated_contributions']),
                    'technical_novelty_count': len(profile['innovation_analysis']['technical_novelty']),
                    'limitations_awareness': len(profile['innovation_analysis']['limitations_acknowledged']),
                    'datasets_expertise': list(profile['innovation_analysis']['datasets_used']),
                    'evaluation_metrics_used': list(profile['innovation_analysis']['evaluation_metrics']),
                    'innovation_level': self._classify_innovation_level(profile['innovation_analysis'])
                },
                'technical_network': {
                    'builds_upon': [{'method': k, 'count': v} for k, v in
                                    profile['technical_relationships']['builds_upon'].most_common(5)],
                    'compared_with': [{'method': k, 'count': v} for k, v in
                                      profile['technical_relationships']['compared_with'].most_common(5)],
                    'extends': [{'theory': k, 'count': v} for k, v in
                                profile['technical_relationships']['extends'].most_common(5)]
                },
                'collaboration_network': {
                    'collaborators': list(profile['collaboration_network']['collaborators']),
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
        """分类研究者产出水平"""
        if paper_count >= 8:
            return "高产研究者"
        elif paper_count >= 5:
            return "活跃研究者"
        elif paper_count >= 3:
            return "稳定研究者"
        elif paper_count >= 2:
            return "新兴研究者"
        else:
            return "初级研究者"

    def _classify_innovation_level(self, innovation_analysis):
        """分类创新水平"""
        contrib_count = len(innovation_analysis['stated_contributions'])
        novelty_count = len(innovation_analysis['technical_novelty'])
        dataset_count = len(innovation_analysis['datasets_used'])

        score = contrib_count * 2 + novelty_count + dataset_count * 0.5

        if score >= 10:
            return "创新引领者"
        elif score >= 6:
            return "创新活跃者"
        elif score >= 3:
            return "创新参与者"
        else:
            return "创新初学者"

    def _classify_collaboration_level(self, collaborator_count):
        """分类合作水平"""
        if collaborator_count >= 10:
            return "合作网络核心"
        elif collaborator_count >= 5:
            return "合作活跃者"
        elif collaborator_count >= 2:
            return "团队合作者"
        else:
            return "独立研究者"

    def analyze_innovation_landscape(self):
        """分析创新格局"""
        print("分析创新格局...")

        innovation_stats = {
            'contribution_types': Counter(),
            'novelty_areas': Counter(),
            'limitation_patterns': Counter(),
            'technical_evolution': defaultdict(list),
            'experimental_trends': {
                'datasets': Counter(),
                'metrics': Counter(),
                'baseline_methods': Counter()
            }
        }

        for doc_id in self.structural_insights:
            structural = self.structural_insights[doc_id]

            # 创新贡献分析
            innovation = structural.get('innovation_analysis', {})
            contributions = self.clean_field_data(innovation.get('stated_contributions', []))
            novelty = self.clean_field_data(innovation.get('stated_novelty', []))

            for contrib in contributions:
                innovation_stats['contribution_types'][contrib] += 1

            for nov in novelty:
                innovation_stats['novelty_areas'][nov] += 1

            # 局限性模式
            limitations = structural.get('limitations', {})
            if limitations:
                ack_limitations = self.clean_field_data(limitations.get('acknowledged_limitations', []))
                for limitation in ack_limitations:
                    innovation_stats['limitation_patterns'][limitation] += 1

            # 实验趋势
            exp_setup = structural.get('experimental_setup', {})
            if exp_setup:
                datasets = exp_setup.get('datasets_used', [])
                metrics = exp_setup.get('evaluation_metrics', [])
                baselines = exp_setup.get('baseline_methods', [])

                for dataset in datasets:
                    if isinstance(dataset, dict):
                        name = dataset.get('dataset_name', '')
                        if name and name != '原文无此信息':
                            innovation_stats['experimental_trends']['datasets'][name] += 1

                for metric in self.clean_field_data(metrics):
                    innovation_stats['experimental_trends']['metrics'][metric] += 1

                for baseline in self.clean_field_data(baselines):
                    innovation_stats['experimental_trends']['baseline_methods'][baseline] += 1

        # 转换为可序列化格式
        landscape_analysis = {
            'contribution_landscape': {
                'total_unique_contributions': len(innovation_stats['contribution_types']),
                'top_contribution_types': [
                    {'contribution': k, 'frequency': v}
                    for k, v in innovation_stats['contribution_types'].most_common(15)
                ]
            },
            'novelty_landscape': {
                'total_novelty_claims': len(innovation_stats['novelty_areas']),
                'top_novelty_areas': [
                    {'area': k, 'frequency': v}
                    for k, v in innovation_stats['novelty_areas'].most_common(15)
                ]
            },
            'limitation_awareness': {
                'total_limitation_types': len(innovation_stats['limitation_patterns']),
                'common_limitations': [
                    {'limitation': k, 'frequency': v}
                    for k, v in innovation_stats['limitation_patterns'].most_common(10)
                ]
            },
            'experimental_ecosystem': {
                'popular_datasets': [
                    {'dataset': k, 'usage_count': v}
                    for k, v in innovation_stats['experimental_trends']['datasets'].most_common(10)
                ],
                'common_metrics': [
                    {'metric': k, 'usage_count': v}
                    for k, v in innovation_stats['experimental_trends']['metrics'].most_common(10)
                ],
                'baseline_methods': [
                    {'method': k, 'usage_count': v}
                    for k, v in innovation_stats['experimental_trends']['baseline_methods'].most_common(10)
                ]
            }
        }

        return landscape_analysis

    def analyze_technical_network(self):
        """分析技术网络和演化"""
        print("🔗 分析技术网络...")

        technical_network = {
            'method_relationships': defaultdict(lambda: {
                'builds_upon': Counter(),
                'compared_with': Counter(),
                'extended_by': Counter()
            }),
            'domain_method_map': defaultdict(set),
            'method_evolution': defaultdict(list)
        }

        for doc_id in self.structural_insights:
            structural = self.structural_insights[doc_id]
            summary = self.summaries.get(doc_id, {})

            # 技术关系
            tech_rel = structural.get('technical_relationships', {})

            # 基于关系
            base_methods = tech_rel.get('base_methods', [])
            for method_info in base_methods:
                if isinstance(method_info, dict):
                    method_name = method_info.get('method_name', '')
                    if method_name and method_name != '原文无此信息':
                        # 这个文档基于这个方法
                        technical_network['method_relationships'][method_name]['extended_by'][doc_id] += 1

            # 对比关系
            compared_methods = tech_rel.get('compared_methods', [])
            current_methods = self.clean_field_data(summary.get('methodology', ''))

            for method_info in compared_methods:
                if isinstance(method_info, dict):
                    method_name = method_info.get('method_name', '')
                    if method_name and method_name != '原文无此信息':
                        for current_method in current_methods:
                            technical_network['method_relationships'][current_method]['compared_with'][method_name] += 1

        # 转换为网络分析结果
        network_analysis = {
            'method_influence_ranking': [],
            'method_comparison_patterns': [],
            'technical_evolution_paths': []
        }

        # 方法影响力排名（被多少论文基于或扩展）
        method_influence = {}
        for method, relationships in technical_network['method_relationships'].items():
            influence_score = len(relationships['extended_by'])
            comparison_count = sum(relationships['compared_with'].values())
            method_influence[method] = {
                'influence_score': influence_score,
                'comparison_count': comparison_count,
                'total_mentions': influence_score + comparison_count
            }

        network_analysis['method_influence_ranking'] = [
            {
                'method': method,
                'influence_score': info['influence_score'],
                'comparison_count': info['comparison_count'],
                'total_mentions': info['total_mentions']
            }
            for method, info in sorted(method_influence.items(),
                                       key=lambda x: x[1]['total_mentions'],
                                       reverse=True)[:20]
        ]

        return network_analysis

    def generate_fusion_visualizations(self, fusion_data):
        """生成融合数据的可视化"""
        print("生成融合可视化...")

        try:
            # 1. 研究者创新水平分布
            author_profiles = fusion_data['deep_author_analysis']
            innovation_levels = Counter()
            for author, profile in author_profiles.items():
                level = profile['innovation_profile']['innovation_level']
                innovation_levels[level] += 1

            if innovation_levels:
                plt.figure(figsize=(12, 8))
                levels = list(innovation_levels.keys())
                counts = list(innovation_levels.values())
                colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']

                plt.pie(counts, labels=levels, autopct='%1.1f%%', colors=colors, startangle=90)
                plt.title('研究者创新水平分布', fontsize=16, fontweight='bold')
                plt.axis('equal')
                plt.tight_layout()
                plt.savefig(VISUALIZATIONS_DIR / 'innovation_level_distribution.png', dpi=300, bbox_inches='tight')
                plt.close()

            # 2. 热门贡献类型柱状图
            contrib_data = fusion_data['innovation_landscape']['contribution_landscape']['top_contribution_types'][:10]
            if contrib_data:
                plt.figure(figsize=(15, 10))
                contributions = [item['contribution'][:50] + '...' if len(item['contribution']) > 50
                                 else item['contribution'] for item in contrib_data]
                frequencies = [item['frequency'] for item in contrib_data]

                plt.barh(contributions, frequencies, color='lightcoral')
                plt.title('热门贡献类型统计', fontsize=16, fontweight='bold')
                plt.xlabel('出现频次')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plt.savefig(VISUALIZATIONS_DIR / 'top_contributions.png', dpi=300, bbox_inches='tight')
                plt.close()

            # 3. 实验生态系统分析
            exp_ecosystem = fusion_data['innovation_landscape']['experimental_ecosystem']

            # 数据集使用情况
            dataset_data = exp_ecosystem['popular_datasets'][:8]
            if dataset_data:
                plt.figure(figsize=(12, 8))
                datasets = [item['dataset'] for item in dataset_data]
                usage_counts = [item['usage_count'] for item in dataset_data]

                plt.bar(datasets, usage_counts, color='skyblue')
                plt.title('热门数据集使用统计', fontsize=16, fontweight='bold')
                plt.ylabel('使用次数')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(VISUALIZATIONS_DIR / 'popular_datasets.png', dpi=300, bbox_inches='tight')
                plt.close()

            # 4. 合作网络强度分布
            collaboration_levels = Counter()
            for author, profile in author_profiles.items():
                level = profile['collaboration_network']['network_level']
                collaboration_levels[level] += 1

            if collaboration_levels:
                plt.figure(figsize=(10, 6))
                levels = list(collaboration_levels.keys())
                counts = list(collaboration_levels.values())
                colors = ['#ffeaa7', '#fab1a0', '#fd79a8', '#e17055']

                plt.bar(levels, counts, color=colors)
                plt.title('研究者合作网络强度分布', fontsize=16, fontweight='bold')
                plt.ylabel('研究者数量')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(VISUALIZATIONS_DIR / 'collaboration_strength.png', dpi=300, bbox_inches='tight')
                plt.close()

            print(f"融合可视化图表已保存到: {VISUALIZATIONS_DIR}")

        except Exception as e:
            print(f"生成可视化时出现错误: {e}")

    def generate_fusion_profiles(self):
        """生成融合学术画像"""
        print("=" * 80)
        print("开始生成融合学术画像")
        print("=" * 80)

        if not self.load_data_sources():
            return None

        # 1. 深度研究者分析
        print("\n" + "=" * 50)
        deep_author_analysis = self.analyze_deep_author_profiles()

        # 2. 创新格局分析
        print("\n" + "=" * 50)
        innovation_landscape = self.analyze_innovation_landscape()

        # 3. 技术网络分析
        print("\n" + "=" * 50)
        technical_network = self.analyze_technical_network()

        # 4. 生成统计信息
        statistics = {
            'data_sources': {
                'summary_documents': len(self.summaries),
                'structural_documents': len(self.structural_insights),
                'total_unique_documents': len(set(self.summaries.keys()) | set(self.structural_insights.keys()))
            },
            'analysis_scope': {
                'total_authors': len(deep_author_analysis),
                'innovation_contributions': len(
                    innovation_landscape['contribution_landscape']['top_contribution_types']),
                'technical_methods': len(technical_network['method_influence_ranking'])
            },
            'generation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # 整合所有分析结果
        fusion_profiles = {
            'metadata': {
                'generation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'data_sources': 'Summary + Structural Insights',
                'analysis_version': '2.0-Fusion',
                'total_documents_analyzed': statistics['data_sources']['total_unique_documents']
            },
            'statistics': statistics,
            'deep_author_analysis': deep_author_analysis,
            'innovation_landscape': innovation_landscape,
            'technical_network': technical_network
        }

        # 保存结果
        with open(FUSION_PROFILES_PATH, 'w', encoding='utf-8') as f:
            json.dump(fusion_profiles, f, indent=2, ensure_ascii=False)

        with open(DEEP_AUTHOR_ANALYSIS_PATH, 'w', encoding='utf-8') as f:
            json.dump(deep_author_analysis, f, indent=2, ensure_ascii=False)

        with open(INNOVATION_LANDSCAPE_PATH, 'w', encoding='utf-8') as f:
            json.dump(innovation_landscape, f, indent=2, ensure_ascii=False)

        with open(TECHNICAL_NETWORK_PATH, 'w', encoding='utf-8') as f:
            json.dump(technical_network, f, indent=2, ensure_ascii=False)

        # 生成可视化
        self.generate_fusion_visualizations(fusion_profiles)

        # 打印摘要
        self.print_fusion_summary(fusion_profiles)

        print(f"\n" + "=" * 80)
        print("融合学术画像生成完成！")
        print(f"完整画像: {FUSION_PROFILES_PATH}")
        print(f"深度作者分析: {DEEP_AUTHOR_ANALYSIS_PATH}")
        print(f"创新格局: {INNOVATION_LANDSCAPE_PATH}")
        print(f"技术网络: {TECHNICAL_NETWORK_PATH}")
        print(f"可视化图表: {VISUALIZATIONS_DIR}")
        print("=" * 80)

        return fusion_profiles

    def print_fusion_summary(self, profiles):
        """打印融合分析摘要"""
        print(f"\n" + "=" * 60)
        print("融合学术画像摘要")
        print("=" * 60)

        stats = profiles['statistics']
        print(f"数据规模:")
        print(f"   概要层文档: {stats['data_sources']['summary_documents']}")
        print(f"   结构化文档: {stats['data_sources']['structural_documents']}")
        print(f"   总计文档: {stats['data_sources']['total_unique_documents']}")

        print(f"\n研究者分析:")
        print(f"   总研究者: {stats['analysis_scope']['total_authors']}")

        # 显示顶级研究者
        author_analysis = profiles['deep_author_analysis']
        top_productive = sorted(author_analysis.items(),
                                key=lambda x: x[1]['basic_info']['total_papers'],
                                reverse=True)[:5]
        print(f" 高产研究者TOP5:")
        for i, (author, profile) in enumerate(top_productive, 1):
            papers = profile['basic_info']['total_papers']
            level = profile['basic_info']['productivity_level']
            print(f"     {i}. {author}: {papers}篇 ({level})")

        print(f"\n创新分析:")
        innovation = profiles['innovation_landscape']
        contrib_count = innovation['contribution_landscape']['total_unique_contributions']
        novelty_count = innovation['novelty_landscape']['total_novelty_claims']
        print(f"   独特贡献类型: {contrib_count}")
        print(f"   新颖性声明: {novelty_count}")

        print(f"   热门贡献TOP3:")
        top_contribs = innovation['contribution_landscape']['top_contribution_types'][:3]
        for i, contrib in enumerate(top_contribs, 1):
            print(f"     {i}. {contrib['contribution'][:50]}... ({contrib['frequency']}次)")

        print(f"\n技术网络:")
        tech_network = profiles['technical_network']
        methods_count = len(tech_network['method_influence_ranking'])
        print(f" 技术方法数: {methods_count}")

        if tech_network['method_influence_ranking']:
            print(f"   影响力最大的方法TOP3:")
            for i, method_info in enumerate(tech_network['method_influence_ranking'][:3], 1):
                method = method_info['method'][:40] + '...' if len(method_info['method']) > 40 else method_info[
                    'method']
                mentions = method_info['total_mentions']
                print(f"     {i}. {method} ({mentions}次提及)")


def main():
    """主函数"""
    generator = FusionAcademicProfileGenerator()
    fusion_profiles = generator.generate_fusion_profiles()
    return fusion_profiles


if __name__ == "__main__":
    main()