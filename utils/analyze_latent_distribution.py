#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_latent_distribution.py
=============================
统计 Interlude 训练数据集中每个样本触发 latent 的次数分布，
并可视化为百分比饼图。

用法:
    python utils/analyze_latent_distribution.py \
        --input /path/to/training_data.json \
        --output ./paper_tables_figures/latent_trigger_distribution.pdf
"""

import argparse
import json
import os
import sys
import numpy as np
from collections import Counter
from typing import Dict, List, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置全局字体
rcParams["font.family"] = ["DejaVu Serif", "serif"]
rcParams["font.serif"] = ["DejaVu Serif", "serif"]


PAUSE_TOKEN = "<|pause|>"

# ============================================================
# 学术论文配色方案 (低饱和度、高可读性)
# 参考 Matplotlib Tableau 风格，适合期刊论文插图
# ============================================================
PIE_COLORS = [
    "#7B9FC3",  # 灰蓝 Academia Blue
    "#78B58C",  # 灰绿 Academia Green
    "#C98B8B",  # 灰粉 Academia Rose
    "#9A8FBF",  # 灰紫 Academia Purple
]


def count_latent_triggers(sample: Dict[str, Any]) -> int:
    """
    计算单个样本中 latent 触发的次数。
    通过统计 reasoning_for_training 中 <|pause|> 的出现次数。
    """
    reasoning = sample.get("reasoning_for_training", "")
    return reasoning.count(PAUSE_TOKEN)


def analyze_distribution(input_path: str):
    """
    读取 JSON 数据集并统计每个样本中 latent 触发次数的分布。
    """
    print(f"正在读取数据集: {input_path}")
    
    # 支持 JSON Lines (.jsonl) 和 JSON array (.json)
    if input_path.endswith(".jsonl"):
        samples = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
    else:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                samples = data
            elif isinstance(data, dict) and "samples" in data:
                samples = data["samples"]
            else:
                raise ValueError(f"不支持的数据格式: {type(data)}")
    
    total_samples = len(samples)
    print(f"数据集总样本数: {total_samples}")
    
    # 统计每个样本的 latent 触发次数
    trigger_counts = [count_latent_triggers(s) for s in samples]
    
    # 分布统计
    dist = Counter(trigger_counts)
    
    print("\nLatent 触发次数分布:")
    print("-" * 40)
    for k in sorted(dist.keys()):
        count = dist[k]
        pct = count / total_samples * 100
        print(f"  触发 {k} 次: {count:>6} 样本 ({pct:>5.2f}%)")
    
    print("-" * 40)
    print(f"  总计: {total_samples} 样本")
    
    # 额外统计
    avg_triggers = sum(trigger_counts) / total_samples
    max_triggers = max(trigger_counts)
    min_triggers = min(trigger_counts)
    print(f"\n平均触发次数: {avg_triggers:.2f}")
    print(f"最大触发次数: {max_triggers}")
    print(f"最小触发次数: {min_triggers}")
    
    return dict(dist), total_samples


def plot_2d_pie_chart(
    dist: Dict[int, int],
    total: int,
    output_path: str,
    figsize: tuple = (10, 8),
):
    """
    绘制普通2D饼图，只标注百分比，使用莫兰迪色系。
    """
    import matplotlib.patches as mpatches
    
    # 按触发次数排序
    labels = sorted(dist.keys())
    counts = [dist[k] for k in labels]
    percentages = [c / total * 100 for c in counts]
    
    # 使用明亮配色
    pie_colors = [PIE_COLORS[i % len(PIE_COLORS)] for i in range(len(labels))]
    
    # 均匀爆炸效果：所有扇形之间均匀分离（减小间隙）
    explode = [0.015 for _ in range(len(labels))]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制2D饼图
    wedges, texts, autotexts = ax.pie(
        counts,
        explode=explode,
        labels=None,  # 不在饼图上直接显示标签
        autopct="%1.1f%%",
        startangle=90,
        colors=pie_colors,
        shadow=False,
        textprops={
            "fontsize": 16,
            "fontweight": "bold",
            "color": "white",
        },
        pctdistance=0.58,
        wedgeprops={
            "edgecolor": "white",
            "linewidth": 2,
        },
    )
    
    # 调整百分比文字位置
    for autotext in autotexts:
        autotext.set_fontsize(18)
        autotext.set_fontweight("bold")
        autotext.set_color("white")
    
    # 标题
    ax.set_title(
        "Distribution of Latent Trigger Counts per Sample",
        fontsize=18,
        fontweight="bold",
        color="#1A252F",
        pad=15,
    )
    
    # 添加图例（显示触发次数和百分比）
    legend_labels = [
        f"{k} Pause{'s' if k != 1 else ''}: {p:.1f}%"
        for k, p in zip(labels, percentages)
    ]
    ax.legend(
        wedges,
        legend_labels,
        title="Latent Triggers",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=14,
        title_fontsize=15,
        frameon=True,
        fancybox=True,
        shadow=False,
        edgecolor="#BDC3C7",
        facecolor="white",
    )
    
    # 确保圆形
    ax.axis("equal")
    
    plt.tight_layout()
    
    # 添加 caption
    fig.text(
        0.5, 0.01,
        "Figure 1: Percentage distribution of latent thinking pause counts per sample in the training dataset.",
        ha="center",
        fontsize=13,
        fontweight="normal",
        color="#333333",
        wrap=True,
    )
    
    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()
    
    print(f"\n2D 饼图已保存至: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="统计 Interlude 训练数据集中 latent 触发次数分布并生成 3D 饼图"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入数据集路径 (JSON 或 JSONL)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./paper_tables_figures/latent_trigger_distribution.pdf",
        help="输出 PDF 路径",
    )
    parser.add_argument(
        "--figsize",
        type=str,
        default="10,8",
        help="图片尺寸 (宽,高)，默认 10,8",
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        sys.exit(1)
    
    figsize = tuple(float(x.strip()) for x in args.figsize.split(","))
    
    dist, total = analyze_distribution(args.input)
    plot_2d_pie_chart(dist, total, args.output, figsize=figsize)


if __name__ == "__main__":
    main()
