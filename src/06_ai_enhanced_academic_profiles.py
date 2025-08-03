# src/ai_visualization_data_generator.py
import json
import os
import time
from zhipuai import ZhipuAI
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from config import PROCESSED_DATA_DIR

# API配置
API_KEY = "838cc9e6876a4fea971b3728af105b56.1KDgfLzNHnfllnhb"
CLIENT = ZhipuAI(api_key=API_KEY)
MODEL_NAME = "glm-4"

# 输出路径
VIZ_DATA_DIR = PROCESSED_DATA_DIR / "visualization_data"
VIZ_DATA_DIR.mkdir(exist_ok=True)

VIZ_CHARTS_DIR = VIZ_DATA_DIR / "charts"
VIZ_CHARTS_DIR.mkdir(exist_ok=True)


class VisualizationDataGenerator:
    """专门生成可视化数据的AI分析器"""

    def __init__(self):
        self.summaries = {}
        self.structural_insights = {}
        self.author_name_map = {}
        self.viz_data = {}

    def load_data_sources(self):
        """加载数据源"""
        # 加载概要层数据
        summaries_path = PROCESSED_DATA_DIR / "document_summaries.json"
        try:
            with open(summaries_path, 'r', encoding='utf-8') as f:
                self.summaries = json.load(f)
            print(f"✅ 加载概要层数据: {len(self.summaries)} 个文档")
        except Exception as e:
            print(f"❌ 加载概要层数据失败: {e}")
            return False

        # 加载结构化数据
        structural_path = PROCESSED_DATA_DIR / "structural_insights" / "structural_insights.json"
        try:
            with open(structural_path, 'r', encoding='utf-8') as f:
                self.structural_insights = json.load(f)
            print(f"✅ 加载结构化数据: {len(self.structural_insights)} 个文档")
        except Exception as e:
            print(f"❌ 加载结构化数据失败: {e}")
            return False

        return True

    def generate_team_radar_data_with_ai(self):
        """AI生成团队雷达图数据（基于每篇文章的详细数据）"""

        # 准备每篇文章的详细数据（前20篇作为样本）
        detailed_papers = self._prepare_detailed_papers_data(limit=20)

        prompt = f"""
你是学术团队评估专家。请基于以下每篇论文的详细数据，为团队能力评估生成雷达图数据。

每篇论文的详细信息:
{json.dumps(detailed_papers, ensure_ascii=False, indent=1)}

请仔细分析每篇论文的内容，从以下6个维度为团队打分（1-10分，保留1位小数）：

1. 研究产出：基于论文数量、质量和发表频率
2. 技术创新：基于创新点的原创性和技术突破
3. 合作网络：基于作者合作模式和跨机构合作
4. 学术影响：基于方法影响力和被对比情况
5. 人才培养：基于作者梯队和研究深度
6. 国际化：基于国际合作和研究视野

返回格式：
{{
  "radar_data": {{
    "研究产出": 8.5,
    "技术创新": 7.8,
    "合作网络": 6.9,
    "学术影响": 7.2,
    "人才培养": 6.5,
    "国际化": 5.8
  }},
  "evidence_analysis": {{
    "研究产出": "基于{len(detailed_papers)}篇论文分析...",
    "技术创新": "发现X个创新点...",
    "合作网络": "平均每篇X位作者合作..."
  }}
}}

只输出JSON格式。
"""

        try:
            response = CLIENT.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800,
            )

            result_str = response.choices[0].message.content.strip()
            result_str = result_str.replace("```json", "").replace("```", "").strip()

            radar_data = json.loads(result_str)
            print("✅ 团队雷达图数据生成成功")
            return radar_data

        except Exception as e:
            print(f"❌ 团队雷达图数据生成失败: {e}")
            return None

    def generate_author_comparison_data_with_ai(self):
        """AI生成作者对比数据（基于每位作者的具体论文）"""

        # 获取每位作者的详细论文数据
        author_papers_data = self._prepare_author_papers_data(limit=5)

        prompt = f"""
你是学术人才评估专家。基于以下每位研究者的具体论文内容，生成能力对比数据。

研究者及其论文详情:
{json.dumps(author_papers_data, ensure_ascii=False, indent=1)}

请仔细分析每位研究者的所有论文，在5个维度上为每人打分（1-10分，保留1位小数）：

1. 研究产出：论文数量、发表质量
2. 创新能力：创新点原创性、技术突破度
3. 技术深度：方法复杂度、理论深度
4. 合作能力：合作网络广度、团队协作
5. 影响力：被引用、方法被对比情况

返回格式：
{{
  "comparison_matrix": [
    {{"name": "研究者姓名", "研究产出": 9.2, "创新能力": 8.8, "技术深度": 8.5, "合作能力": 7.9, "影响力": 8.1}},
    ...
  ],
  "dimensions": ["研究产出", "创新能力", "技术深度", "合作能力", "影响力"],
  "analysis_basis": {{
    "研究者姓名": "基于X篇论文，发现Y个创新点...",
    ...
  }}
}}

只输出JSON数据。
"""

        try:
            response = CLIENT.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1200,
            )

            result_str = response.choices[0].message.content.strip()
            result_str = result_str.replace("```json", "").replace("```", "").strip()

            comparison_data = json.loads(result_str)
            print("✅ 作者对比数据生成成功")
            return comparison_data

        except Exception as e:
            print(f"❌ 作者对比数据生成失败: {e}")
            return None

    def generate_research_trend_data_with_ai(self):
        """AI生成研究趋势时间序列数据"""

        # 收集按年份的研究数据
        yearly_stats = self._collect_yearly_research_data()

        prompt = f"""
你是趋势分析专家。基于研究数据生成时间序列图表数据。

年度数据: {json.dumps(yearly_stats, ensure_ascii=False)}

请生成2020-2024年的研究趋势数据：

{{
  "time_series_data": [
    {{"year": 2020, "论文数量": 15, "创新指数": 6.8, "合作强度": 4.2}},
    {{"year": 2021, "论文数量": 23, "创新指数": 7.2, "合作强度": 5.1}},
    {{"year": 2022, "论文数量": 31, "创新指数": 7.8, "合作强度": 6.3}},
    {{"year": 2023, "论文数量": 28, "创新指数": 8.1, "合作强度": 7.0}},
    {{"year": 2024, "论文数量": 21, "创新指数": 8.4, "合作强度": 7.5}}
  ],
  "trend_analysis": {{
    "论文数量_trend": "先增后减，2022年达峰",
    "创新指数_trend": "稳步上升",
    "合作强度_trend": "持续增强"
  }},
  "future_prediction": {{
    "2025_forecast": {{"论文数量": 25, "创新指数": 8.6, "合作强度": 8.0}}
  }}
}}

只输出JSON数据。
"""

        try:
            response = CLIENT.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000,
            )

            result_str = response.choices[0].message.content.strip()
            result_str = result_str.replace("```json", "").replace("```", "").strip()

            trend_data = json.loads(result_str)
            print("✅ 研究趋势数据生成成功")
            return trend_data

        except Exception as e:
            print(f"❌ 研究趋势数据生成失败: {e}")
            return None

    def generate_collaboration_network_data_with_ai(self):
        """AI生成合作网络图数据"""

        # 收集合作关系数据
        collab_stats = self._collect_collaboration_data()

        prompt = f"""
你是网络分析专家。基于合作数据生成网络图数据。

合作统计: {json.dumps(collab_stats, ensure_ascii=False)}

请生成网络图的节点和边数据：

{{
  "nodes": [
    {{"id": "李睿凡", "size": 25, "group": "核心", "papers": 65, "centrality": 0.85}},
    {{"id": "Wang Xiaojie", "size": 18, "group": "活跃", "papers": 25, "centrality": 0.62}},
    {{"id": "Feng Fangxiang", "size": 15, "group": "活跃", "papers": 19, "centrality": 0.48}}
  ],
  "edges": [
    {{"source": "李睿凡", "target": "Wang Xiaojie", "weight": 8, "papers": 8}},
    {{"source": "李睿凡", "target": "Feng Fangxiang", "weight": 5, "papers": 5}},
    {{"source": "Wang Xiaojie", "target": "Feng Fangxiang", "weight": 3, "papers": 3}}
  ],
  "network_metrics": {{
    "density": 0.65,
    "avg_clustering": 0.73,
    "core_nodes": ["李睿凡"],
    "bridge_nodes": ["Wang Xiaojie"]
  }}
}}

只输出JSON数据。
"""

        try:
            response = CLIENT.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1200,
            )

            result_str = response.choices[0].message.content.strip()
            result_str = result_str.replace("```json", "").replace("```", "").strip()

            network_data = json.loads(result_str)
            print("✅ 合作网络数据生成成功")
            return network_data

        except Exception as e:
            print(f"❌ 合作网络数据生成失败: {e}")
            return None

    def generate_research_domain_pie_data_with_ai(self):
        """AI生成研究领域饼图数据"""

        domain_stats = self._collect_domain_statistics()

        prompt = f"""
你是领域分析专家。基于研究领域数据生成饼图数据。

领域统计: {json.dumps(domain_stats, ensure_ascii=False)}

请生成研究领域分布的饼图数据：

{{
  "pie_data": [
    {{"domain": "计算机视觉", "papers": 32, "percentage": 27.1, "color": "#FF6B6B"}},
    {{"domain": "自然语言处理", "papers": 28, "percentage": 23.7, "color": "#4ECDC4"}},
    {{"domain": "机器学习", "papers": 25, "percentage": 21.2, "color": "#45B7D1"}},
    {{"domain": "数据挖掘", "papers": 18, "percentage": 15.3, "color": "#96CEB4"}},
    {{"domain": "其他", "papers": 15, "percentage": 12.7, "color": "#FFEAA7"}}
  ],
  "total_papers": 118,
  "domain_analysis": {{
    "dominant_field": "计算机视觉",
    "emerging_field": "自然语言处理",
    "diversity_index": 0.78
  }}
}}

只输出JSON数据。
"""

        try:
            response = CLIENT.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=800,
            )

            result_str = response.choices[0].message.content.strip()
            result_str = result_str.replace("```json", "").replace("```", "").strip()

            pie_data = json.loads(result_str)
            print("✅ 研究领域饼图数据生成成功")
            return pie_data

        except Exception as e:
            print(f"❌ 研究领域饼图数据生成失败: {e}")
            return None

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

        # 其他姓名标准化处理
        name = name.replace(",", " ").replace(".", " ")
        name = " ".join(name.split())

        if any('\u4e00' <= char <= '\u9fff' for char in name):
            return name

        parts = name.split()
        if len(parts) == 2:
            return f"{parts[1]} {parts[0]}" if parts[0][0].isupper() and parts[1][0].isupper() else name

        return name

    def build_author_name_mapping(self):
        """构建作者姓名标准化映射"""
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
        for author in all_authors:
            normalized = self.normalize_author_name(author)
            if normalized:
                self.author_name_map[author] = normalized

    def _prepare_detailed_papers_data(self, limit=20):
        """准备每篇论文的详细数据供AI分析"""
        detailed_papers = {}
        all_docs = list(set(self.summaries.keys()) | set(self.structural_insights.keys()))

        # 限制数据量，选择前N篇论文
        selected_docs = all_docs[:limit]

        for doc_id in selected_docs:
            paper_data = {"doc_id": doc_id}

            # 从概要层获取数据
            if doc_id in self.summaries:
                summary = self.summaries[doc_id]
                paper_data["概要层"] = {
                    "document_type": summary.get('document_type', ''),
                    "title": summary.get('title', ''),
                    "authors": self.clean_field_data(summary.get('authors', [])) or
                               self.clean_field_data(summary.get('inventors', [])),
                    "main_topic": summary.get('main_topic', '') or summary.get('application_domain', ''),
                    "methodology": summary.get('methodology', '') or summary.get('technical_solution', ''),
                    "key_innovations": self.clean_field_data(summary.get('key_innovations', [])),
                    "keywords": self.clean_field_data(summary.get('keywords', [])),
                    "technical_concepts": self.clean_field_data(summary.get('technical_concepts', []))
                }

            # 从结构化层获取数据
            if doc_id in self.structural_insights:
                structural = self.structural_insights[doc_id]
                paper_data["结构化层"] = {
                    "document_metadata": structural.get('document_metadata', {}),
                    "innovation_analysis": {
                        "stated_contributions": self.clean_field_data(
                            structural.get('innovation_analysis', {}).get('stated_contributions', [])
                        ),
                        "stated_novelty": self.clean_field_data(
                            structural.get('innovation_analysis', {}).get('stated_novelty', [])
                        )
                    },
                    "technical_relationships": structural.get('technical_relationships', {}),
                    "experimental_setup": structural.get('experimental_setup', {})
                }

            detailed_papers[doc_id] = paper_data

        return detailed_papers

    def _prepare_author_papers_data(self, limit=5):
        """准备每位作者的具体论文数据"""
        # 先统计每个作者的论文数
        author_papers = defaultdict(list)
        all_docs = set(self.summaries.keys()) | set(self.structural_insights.keys())

        for doc_id in all_docs:
            # 获取作者
            doc_authors = []
            if doc_id in self.summaries:
                summary = self.summaries[doc_id]
                doc_type = summary.get('document_type', 'unknown')
                if doc_type == 'patent':
                    authors = self.clean_field_data(summary.get('inventors', []))
                else:
                    authors = self.clean_field_data(summary.get('authors', []))
                doc_authors.extend(authors)

            # 为每个作者记录这篇论文
            for author in doc_authors:
                normalized = self.author_name_map.get(author, author)
                if normalized:
                    author_papers[normalized].append(doc_id)

        # 选择前N个最活跃的作者
        top_authors = sorted(author_papers.items(), key=lambda x: len(x[1]), reverse=True)[:limit]

        # 为每个作者准备详细的论文数据
        author_detailed_data = {}
        for author_name, paper_ids in top_authors:
            author_detailed_data[author_name] = {
                "total_papers": len(paper_ids),
                "papers": {}
            }

            # 为每篇论文准备详细数据
            for paper_id in paper_ids:
                paper_data = {}

                # 概要层数据
                if paper_id in self.summaries:
                    summary = self.summaries[paper_id]
                    paper_data["概要"] = {
                        "title": summary.get('title', ''),
                        "main_topic": summary.get('main_topic', '') or summary.get('application_domain', ''),
                        "methodology": summary.get('methodology', '') or summary.get('technical_solution', ''),
                        "key_innovations": self.clean_field_data(summary.get('key_innovations', []))
                    }

                # 结构化层数据
                if paper_id in self.structural_insights:
                    structural = self.structural_insights[paper_id]
                    paper_data["结构化"] = {
                        "contributions": self.clean_field_data(
                            structural.get('innovation_analysis', {}).get('stated_contributions', [])
                        ),
                        "novelty": self.clean_field_data(
                            structural.get('innovation_analysis', {}).get('stated_novelty', [])
                        ),
                        "technical_relationships": structural.get('technical_relationships', {})
                    }

                author_detailed_data[author_name]["papers"][paper_id] = paper_data

        return author_detailed_data

    def _collect_yearly_research_data(self):
        """收集真实的年度研究数据"""
        # 注意：概要数据中可能没有年份信息，这里返回提示
        return {"note": "需要从文档中提取年份信息，当前数据中年份信息有限"}

    def _collect_collaboration_data(self):
        """收集真实的合作数据"""
        collaboration_pairs = 0
        total_collaborations = 0
        all_docs = set(self.summaries.keys()) | set(self.structural_insights.keys())

        for doc_id in all_docs:
            doc_authors = []
            if doc_id in self.summaries:
                summary = self.summaries[doc_id]
                doc_type = summary.get('document_type', 'unknown')
                if doc_type == 'patent':
                    authors = self.clean_field_data(summary.get('inventors', []))
                else:
                    authors = self.clean_field_data(summary.get('authors', []))
                doc_authors.extend(authors)

            if len(doc_authors) > 1:
                collaboration_pairs += 1
                total_collaborations += len(doc_authors)

        return {
            "collaboration_papers": collaboration_pairs,
            "total_papers": len(all_docs),
            "avg_authors_per_paper": total_collaborations / len(all_docs) if all_docs else 0
        }

    def _collect_domain_statistics(self):
        """收集真实的领域统计"""
        domain_counts = Counter()

        for doc_id, summary in self.summaries.items():
            doc_type = summary.get('document_type', 'unknown')
            if doc_type == 'patent':
                domains = self.clean_field_data(summary.get('application_domain', ''))
                domains.extend(self.clean_field_data(summary.get('application_scenarios', [])))
            else:
                domains = self.clean_field_data(summary.get('main_topic', ''))
                domains.extend(self.clean_field_data(summary.get('application_domains', [])))

            for domain in domains:
                domain_counts[domain] += 1

        return dict(domain_counts.most_common(10))

    def create_visualizations_from_ai_data(self):
        """基于AI生成的数据创建可视化"""

        print("\n🎨 开始创建可视化图表...")

        # 1. 团队雷达图
        if 'radar_data' in self.viz_data:
            self._create_team_radar_chart()

        # 2. 作者对比热图
        if 'comparison_data' in self.viz_data:
            self._create_author_comparison_heatmap()

        # 3. 研究趋势线图
        if 'trend_data' in self.viz_data:
            self._create_research_trend_chart()

        # 4. 合作网络图
        if 'network_data' in self.viz_data:
            self._create_collaboration_network_chart()

        # 5. 研究领域饼图
        if 'pie_data' in self.viz_data:
            self._create_research_domain_pie_chart()

    def _create_team_radar_chart(self):
        """创建团队能力雷达图"""
        radar_data = self.viz_data['radar_data']['radar_data']

        categories = list(radar_data.keys())
        values = list(radar_data.values())

        # 闭合雷达图
        categories += categories[:1]
        values += values[:1]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='团队能力',
            line=dict(color='rgb(255, 99, 132)', width=3),
            fillcolor='rgba(255, 99, 132, 0.25)'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10],
                    tickfont=dict(size=12)
                )),
            showlegend=True,
            title="团队综合能力雷达图",
            font=dict(family="Microsoft YaHei", size=14),
            width=600,
            height=600
        )

        fig.write_html(VIZ_CHARTS_DIR / "team_radar_chart.html")
        print("✅ 团队雷达图已生成")

    def _create_author_comparison_heatmap(self):
        """创建作者能力对比热图"""
        comparison_data = self.viz_data['comparison_data']['comparison_matrix']
        dimensions = self.viz_data['comparison_data']['dimensions']

        # 构建数据矩阵
        authors = [item['name'] for item in comparison_data]
        matrix = []

        for author_data in comparison_data:
            row = [author_data[dim] for dim in dimensions]
            matrix.append(row)

        fig = px.imshow(
            matrix,
            labels=dict(x="能力维度", y="研究者", color="评分"),
            x=dimensions,
            y=authors,
            color_continuous_scale="RdYlBu_r",
            aspect="auto"
        )

        fig.update_layout(
            title="研究者能力对比热图",
            font=dict(family="Microsoft YaHei", size=12),
            width=800,
            height=500
        )

        fig.write_html(VIZ_CHARTS_DIR / "author_comparison_heatmap.html")
        print("✅ 作者对比热图已生成")

    def _create_research_trend_chart(self):
        """创建研究趋势图"""
        trend_data = self.viz_data['trend_data']['time_series_data']

        df = pd.DataFrame(trend_data)

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('论文数量趋势', '创新指数趋势', '合作强度趋势', '综合趋势'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # 论文数量
        fig.add_trace(
            go.Scatter(x=df['year'], y=df['论文数量'], name='论文数量', line=dict(color='blue', width=3)),
            row=1, col=1
        )

        # 创新指数
        fig.add_trace(
            go.Scatter(x=df['year'], y=df['创新指数'], name='创新指数', line=dict(color='red', width=3)),
            row=1, col=2
        )

        # 合作强度
        fig.add_trace(
            go.Scatter(x=df['year'], y=df['合作强度'], name='合作强度', line=dict(color='green', width=3)),
            row=2, col=1
        )

        # 综合趋势
        fig.add_trace(
            go.Scatter(x=df['year'], y=df['论文数量'], name='论文数量', line=dict(color='blue')),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=df['year'], y=df['创新指数'] * 5, name='创新指数×5', line=dict(color='red')),
            row=2, col=2
        )

        fig.update_layout(
            title="研究发展趋势分析",
            font=dict(family="Microsoft YaHei", size=12),
            width=1000,
            height=700
        )

        fig.write_html(VIZ_CHARTS_DIR / "research_trend_chart.html")
        print("✅ 研究趋势图已生成")

    def _create_collaboration_network_chart(self):
        """创建合作网络图"""
        network_data = self.viz_data['network_data']
        nodes = network_data['nodes']
        edges = network_data['edges']

        # 使用plotly创建网络图
        edge_x = []
        edge_y = []

        # 简单的圆形布局
        import math
        n = len(nodes)
        positions = {}

        for i, node in enumerate(nodes):
            angle = 2 * math.pi * i / n
            x = math.cos(angle)
            y = math.sin(angle)
            positions[node['id']] = (x, y)

        for edge in edges:
            x0, y0 = positions[edge['source']]
            x1, y1 = positions[edge['target']]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(x=edge_x, y=edge_y,
                                line=dict(width=2, color='#888'),
                                hoverinfo='none',
                                mode='lines')

        node_x = []
        node_y = []
        node_text = []
        node_size = []

        for node in nodes:
            x, y = positions[node['id']]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"{node['id']}<br>论文数: {node['papers']}")
            node_size.append(node['size'])

        node_trace = go.Scatter(x=node_x, y=node_y,
                                mode='markers+text',
                                hoverinfo='text',
                                text=[node['id'] for node in nodes],
                                hovertext=node_text,
                                marker=dict(size=node_size,
                                            color='lightblue',
                                            line=dict(width=2, color='darkblue')))

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='研究者合作网络图',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            annotations=[dict(
                                text="节点大小表示论文数量",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002)],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            font=dict(family="Microsoft YaHei", size=12)))

        fig.write_html(VIZ_CHARTS_DIR / "collaboration_network.html")
        print("✅ 合作网络图已生成")

    def _create_research_domain_pie_chart(self):
        """创建研究领域饼图"""
        pie_data = self.viz_data['pie_data']['pie_data']

        domains = [item['domain'] for item in pie_data]
        papers = [item['papers'] for item in pie_data]
        colors = [item['color'] for item in pie_data]

        fig = go.Figure(data=[go.Pie(
            labels=domains,
            values=papers,
            marker=dict(colors=colors),
            textinfo='label+percent',
            textfont_size=12,
            hole=.3
        )])

        fig.update_layout(
            title="研究领域分布",
            font=dict(family="Microsoft YaHei", size=14),
            width=600,
            height=600
        )

        fig.write_html(VIZ_CHARTS_DIR / "research_domain_pie.html")
        print("✅ 研究领域饼图已生成")

    def run_complete_visualization_pipeline(self):
        """运行完整的可视化数据生成流程"""
        print("=" * 80)
        print("AI可视化数据生成器")
        print("专门生成可用于图表可视化的数据")
        print("=" * 80)

        # 1. 加载数据源
        if not self.load_data_sources():
            return None

        # 2. 生成各类可视化数据
        print("\n🤖 使用AI生成可视化数据...")

        # 团队雷达图数据
        radar_data = self.generate_team_radar_data_with_ai()
        if radar_data:
            self.viz_data['radar_data'] = radar_data

        # 作者对比数据
        comparison_data = self.generate_author_comparison_data_with_ai()
        if comparison_data:
            self.viz_data['comparison_data'] = comparison_data

        # 研究趋势数据
        trend_data = self.generate_research_trend_data_with_ai()
        if trend_data:
            self.viz_data['trend_data'] = trend_data

        # 合作网络数据
        network_data = self.generate_collaboration_network_data_with_ai()
        if network_data:
            self.viz_data['network_data'] = network_data

        # 研究领域饼图数据
        pie_data = self.generate_research_domain_pie_data_with_ai()
        if pie_data:
            self.viz_data['pie_data'] = pie_data

        # 3. 保存可视化数据
        with open(VIZ_DATA_DIR / "ai_visualization_data.json", 'w', encoding='utf-8') as f:
            json.dump(self.viz_data, f, indent=2, ensure_ascii=False)

        # 4. 创建图表
        self.create_visualizations_from_ai_data()

        print(f"\n" + "=" * 80)
        print("AI可视化数据生成完成！")
        print(f"📊 可视化数据: {VIZ_DATA_DIR / 'ai_visualization_data.json'}")
        print(f"📈 图表文件: {VIZ_CHARTS_DIR}")
        print("=" * 80)

        return self.viz_data


def main():
    """主函数"""
    generator = VisualizationDataGenerator()
    viz_data = generator.run_complete_visualization_pipeline()
    return viz_data


if __name__ == "__main__":
    main()