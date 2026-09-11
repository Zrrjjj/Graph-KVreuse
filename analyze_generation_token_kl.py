#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_generation_token_kl.py

分析完整生成过程中的 reference_token_text 与 KL 分布。

输入：
    per_generation_step.csv

该文件由：
    benchmark_cacheclip_generation_kl.py

生成。

核心字段：
    sample_id
    method
    repair_ratio
    generation_step
    history_length_before_token
    reference_token_id
    reference_token_text
    kl
    js
    logits_cosine
    top1_match
    topk_overlap
    full_top1_probability
    method_probability_on_full_top1

输出：
    01_top_kl_generation_tokens.csv
    02_generation_token_summary.csv
    03_generation_step_summary.csv
    04_generation_token_type_summary.csv
    05_generation_token_kl_matrix.csv
    06_analysis_metadata.json

图像：
    01_generation_step_kl_by_token.png/pdf
    02_generation_step_kl_heatmap.png/pdf
    03_token_type_kl_distribution.png/pdf
    04_high_kl_token_distribution.png/pdf
    05_topk_and_cosine_by_step.png/pdf
    06_token_frequency_vs_kl.png/pdf

实验含义：
    当前数据来自 teacher-forced generation trajectory。
    reference_token_text 是 Full Recompute 参考轨迹中
    当前 generation step 对应的 token。
"""

import re
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns


# ============================================================
# 1. 论文风格
# ============================================================

METHOD_COLORS = {
    "Full Long-Path KV Reuse": "#D95F02",
    "CacheClip Token Repair 10%": "#8DA0CB",
    "CacheClip Token Repair 20%": "#7570B3",
    "CacheClip Token Repair 30%": "#4C78A8",
    "CacheClip Token Repair 40%": "#1B9E77",
}

METHOD_MARKERS = {
    "Full Long-Path KV Reuse": "D",
    "CacheClip Token Repair 10%": "o",
    "CacheClip Token Repair 20%": "s",
    "CacheClip Token Repair 30%": "^",
    "CacheClip Token Repair 40%": "P",
}


def setup_paper_style():
    sns.set_theme(
        style="whitegrid",
        context="paper",
    )

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 420,
            "savefig.bbox": "tight",

            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "DejaVu Serif",
                "Liberation Serif",
            ],

            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,

            "axes.linewidth": 1.0,
            "lines.linewidth": 2.2,
            "lines.markersize": 6.5,

            "grid.alpha": 0.25,
            "grid.linestyle": "--",

            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig, output_dir, filename):
    png_path = output_dir / f"{filename}.png"
    pdf_path = output_dir / f"{filename}.pdf"

    fig.savefig(
        png_path,
        dpi=420,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"[Saved] {png_path}")
    print(f"[Saved] {pdf_path}")


# ============================================================
# 2. 方法名称与颜色
# ============================================================

def get_method_color(method):
    if method in METHOD_COLORS:
        return METHOD_COLORS[method]

    if "CacheClip" in str(method):
        return "#7570B3"

    return "#555555"


def get_method_marker(method):
    if method in METHOD_MARKERS:
        return METHOD_MARKERS[method]

    if "CacheClip" in str(method):
        return "o"

    return "o"


def method_order(method):
    method = str(method)

    if method == "Full Long-Path KV Reuse":
        return 0.0

    match = re.search(
        r"CacheClip Token Repair\s+(\d+)%",
        method,
    )

    if match:
        return 1.0 + int(match.group(1)) / 100.0

    return 99.0


# ============================================================
# 3. Token 文本处理
# ============================================================

def clean_token_text(text):
    """
    将 CSV 中的 token 文本转换为便于阅读的形式。
    """

    if pd.isna(text):
        return "<EMPTY>"

    text = str(text)

    text = text.replace(
        "\n",
        "\\n",
    )

    text = text.replace(
        "\r",
        "\\r",
    )

    if text.strip() == "":
        return "<SPACE>"

    return text


def classify_reference_token(
    token_text,
    token_id=None,
    eos_token_id=None,
):
    """
    将 reference_token_text 分为：

        entity_like
        digit
        underscore
        punctuation
        newline
        whitespace
        eos_like
        other

    这是启发式分类，主要用于诊断和可视化。
    """

    if token_text is None or pd.isna(token_text):
        return "other"

    text = str(token_text)

    if (
        eos_token_id is not None
        and token_id is not None
    ):
        try:
            if int(token_id) == int(eos_token_id):
                return "eos_like"
        except Exception:
            pass

    if "\\n" in text or "\n" in text:
        return "newline"

    if text.strip() == "":
        return "whitespace"

    # 常见 entity ID，例如 E_006、E_12。
    if re.search(
        r"[Ee][\s_\-]*\d+",
        text,
    ):
        return "entity_like"

    if text.strip().isdigit():
        return "digit"

    if "_" in text:
        return "underscore"

    if all(
        char in ".,;:!?()[]{}<>\"'`-+/=|"
        for char in text
    ):
        return "punctuation"

    return "other"


# ============================================================
# 4. 数据读取
# ============================================================

def load_generation_step_data(
    input_csv,
    eos_token_id=None,
):
    input_csv = Path(input_csv)

    if not input_csv.exists():
        raise FileNotFoundError(
            f"Cannot find:\n{input_csv}"
        )

    df = pd.read_csv(input_csv)

    required_columns = {
        "sample_id",
        "method",
        "generation_step",
        "reference_token_text",
        "kl",
        "js",
        "logits_cosine",
        "top1_match",
        "topk_overlap",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Input CSV missing columns:\n{missing}"
        )

    if "repair_ratio" not in df.columns:
        df["repair_ratio"] = np.nan

    if "reference_token_id" not in df.columns:
        df["reference_token_id"] = np.nan

    numeric_columns = [
        "generation_step",
        "repair_ratio",
        "reference_token_id",
        "kl",
        "js",
        "logits_cosine",
        "top1_match",
        "topk_overlap",
        "full_top1_probability",
        "method_probability_on_full_top1",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    df["reference_token_text_clean"] = (
        df["reference_token_text"]
        .apply(clean_token_text)
    )

    df["token_type"] = df.apply(
        lambda row: classify_reference_token(
            token_text=row[
                "reference_token_text"
            ],
            token_id=row.get(
                "reference_token_id",
                None,
            ),
            eos_token_id=eos_token_id,
        ),
        axis=1,
    )

    df["method_order"] = df["method"].apply(
        method_order
    )

    df = df.sort_values(
        [
            "method_order",
            "generation_step",
            "sample_id",
        ]
    ).reset_index(drop=True)

    return df


# ============================================================
# 5. Bootstrap CI
# ============================================================

def bootstrap_mean_ci(
    values,
    n_bootstrap=3000,
    confidence=0.95,
    seed=42,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[~np.isnan(values)]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    mean_value = float(values.mean())

    if len(values) == 1:
        return mean_value, mean_value, mean_value

    rng = np.random.default_rng(seed)

    boot_means = np.empty(
        n_bootstrap,
        dtype=np.float64,
    )

    for index in range(n_bootstrap):
        sampled = rng.choice(
            values,
            size=len(values),
            replace=True,
        )

        boot_means[index] = sampled.mean()

    alpha = 1.0 - confidence

    low = float(
        np.quantile(
            boot_means,
            alpha / 2.0,
        )
    )

    high = float(
        np.quantile(
            boot_means,
            1.0 - alpha / 2.0,
        )
    )

    return mean_value, low, high


# ============================================================
# 6. Step-level 汇总
# ============================================================

def summarize_by_generation_step(
    df,
    n_bootstrap,
):
    """
    对 method × generation_step 聚合。
    """

    rows = []

    for (
        method,
        repair_ratio,
        generation_step,
    ), group in df.groupby(
        [
            "method",
            "repair_ratio",
            "generation_step",
        ],
        dropna=False,
    ):
        row = {
            "method": method,
            "repair_ratio": repair_ratio,
            "generation_step": int(
                generation_step
            ),
            "n_rows": len(group),
            "n_samples": group[
                "sample_id"
            ].nunique(),
        }

        for metric in [
            "kl",
            "js",
            "logits_cosine",
            "top1_match",
            "topk_overlap",
            "full_top1_probability",
            "method_probability_on_full_top1",
        ]:
            if metric not in group.columns:
                continue

            mean_value, low, high = (
                bootstrap_mean_ci(
                    group[metric].to_numpy(),
                    n_bootstrap=n_bootstrap,
                )
            )

            row[f"mean_{metric}"] = mean_value
            row[f"ci_low_{metric}"] = low
            row[f"ci_high_{metric}"] = high

        rows.append(row)

    result = pd.DataFrame(rows)

    if len(result) == 0:
        return result

    result["method_order"] = result[
        "method"
    ].apply(method_order)

    return result.sort_values(
        [
            "method_order",
            "generation_step",
        ]
    ).reset_index(drop=True)


# ============================================================
# 7. Token-level Summary
# ============================================================

def summarize_by_reference_token(
    df,
):
    """
    按 method × generation_step × reference_token_text 汇总。

    用于找出：
        某个具体生成 token 出现时，
        KL 是否特别高。
    """

    grouped = (
        df.groupby(
            [
                "method",
                "repair_ratio",
                "generation_step",
                "reference_token_id",
                "reference_token_text_clean",
                "token_type",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            n_rows=(
                "sample_id",
                "count",
            ),
            n_samples=(
                "sample_id",
                "nunique",
            ),
            mean_kl=(
                "kl",
                "mean",
            ),
            median_kl=(
                "kl",
                "median",
            ),
            p90_kl=(
                "kl",
                lambda x: x.quantile(0.90),
            ),
            mean_js=(
                "js",
                "mean",
            ),
            mean_logits_cosine=(
                "logits_cosine",
                "mean",
            ),
            mean_top1_match=(
                "top1_match",
                "mean",
            ),
            mean_topk_overlap=(
                "topk_overlap",
                "mean",
            ),
        )
    )

    grouped["method_order"] = grouped[
        "method"
    ].apply(method_order)

    return grouped.sort_values(
        [
            "method_order",
            "mean_kl",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(drop=True)


# ============================================================
# 8. Token Type Summary
# ============================================================

def summarize_by_token_type(
    df,
):
    return (
        df.groupby(
            [
                "method",
                "repair_ratio",
                "token_type",
            ],
            as_index=False,
        )
        .agg(
            n_rows=(
                "sample_id",
                "count",
            ),
            n_samples=(
                "sample_id",
                "nunique",
            ),
            mean_kl=(
                "kl",
                "mean",
            ),
            median_kl=(
                "kl",
                "median",
            ),
            p90_kl=(
                "kl",
                lambda x: x.quantile(0.90),
            ),
            mean_js=(
                "js",
                "mean",
            ),
            mean_logits_cosine=(
                "logits_cosine",
                "mean",
            ),
            mean_topk_overlap=(
                "topk_overlap",
                "mean",
            ),
        )
    )


# ============================================================
# 9. 具体高 KL 行
# ============================================================

def extract_top_kl_rows(
    df,
    top_n=100,
):
    """
    提取所有方法中 KL 最大的具体行。
    """

    columns = [
        "sample_id",
        "method",
        "repair_ratio",
        "generation_step",
        "reference_token_id",
        "reference_token_text",
        "reference_token_text_clean",
        "token_type",
        "kl",
        "js",
        "logits_cosine",
        "top1_match",
        "topk_overlap",
        "full_top1_probability",
        "method_probability_on_full_top1",
    ]

    columns = [
        column
        for column in columns
        if column in df.columns
    ]

    return df.sort_values(
        "kl",
        ascending=False,
    )[columns].head(top_n).reset_index(drop=True)


# ============================================================
# 10. Figure 1：Generation Step KL Curve
# ============================================================

def plot_step_kl_curve(
    step_df,
    output_dir,
):
    fig, ax = plt.subplots(
        figsize=(11.0, 6.3),
    )

    methods = sorted(
        step_df["method"].unique(),
        key=method_order,
    )

    for method in methods:
        subset = step_df[
            step_df["method"] == method
        ].sort_values(
            "generation_step"
        )

        if len(subset) == 0:
            continue

        color = get_method_color(method)
        marker = get_method_marker(method)

        x = subset[
            "generation_step"
        ].to_numpy()

        y = subset[
            "mean_kl"
        ].to_numpy()

        low = subset[
            "ci_low_kl"
        ].to_numpy()

        high = subset[
            "ci_high_kl"
        ].to_numpy()

        ax.plot(
            x,
            y,
            color=color,
            marker=marker,
            label=method,
            zorder=4,
        )

        ax.fill_between(
            x,
            np.maximum(0, low),
            high,
            color=color,
            alpha=0.13,
            linewidth=0,
        )

    ax.set_xlabel(
        "Generation Step along Full-Recompute Reference Trajectory"
    )

    ax.set_ylabel(
        r"Mean Step-wise KL "
        r"$D_{\mathrm{KL}}(P_{\mathrm{Full}}\parallel P_{\mathrm{Method}})$"
    )

    ax.set_title(
        "Generation-Step KL Divergence of CacheClip KV Reuse",
        pad=12,
        fontweight="bold",
    )

    # KL 可能跨越较大数量级。
    ax.set_yscale("log")

    ax.set_xlim(
        0.5,
        max(
            1.5,
            step_df["generation_step"].max() + 0.5,
        ),
    )

    ax.grid(
        axis="both",
        alpha=0.25,
    )

    sns.despine(ax=ax)

    ax.legend(
        loc="best",
        frameon=True,
        fontsize=8.8,
    )

    save_figure(
        fig,
        output_dir,
        "01_generation_step_kl_curve",
    )


# ============================================================
# 11. Figure 2：Generation Step KL Heatmap
# ============================================================

def plot_step_kl_heatmap(
    step_df,
    output_dir,
):
    """
    行：
        method

    列：
        generation step

    值：
        mean KL
    """

    methods = sorted(
        step_df["method"].unique(),
        key=method_order,
    )

    matrix = step_df.pivot(
        index="method",
        columns="generation_step",
        values="mean_kl",
    )

    matrix = matrix.reindex(
        methods
    )

    matrix.index = [
        method.replace(
            "CacheClip Token Repair ",
            "CacheClip ",
        )
        for method in matrix.index
    ]

    fig, ax = plt.subplots(
        figsize=(13.5, 5.6),
    )

    # 使用 log10 颜色尺度，避免 Step 8/10/11 完全压扁其他位置。
    values = matrix.to_numpy(
        dtype=np.float64
    )

    safe_values = np.maximum(
        values,
        1e-8,
    )

    log_values = np.log10(
        safe_values
    )

    sns.heatmap(
        log_values,
        annot=values,
        fmt=".2f",
        cmap="magma",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={
            "label": r"$\log_{10}$(Mean KL)"
        },
        ax=ax,
    )

    ax.set_title(
        "Generation-Step KL Heatmap",
        pad=12,
        fontweight="bold",
    )

    ax.set_xlabel("Generation Step")
    ax.set_ylabel("Method")

    save_figure(
        fig,
        output_dir,
        "02_generation_step_kl_heatmap",
    )


# ============================================================
# 12. Figure 3：Token Type KL Distribution
# ============================================================

def plot_token_type_kl_distribution(
    df,
    output_dir,
):
    """
    展示不同 reference token 类型对应的 KL 分布。

    使用所有 CacheClip 方法，
    不包括 Full KV Reuse。
    """

    cacheclip_df = df[
        df["method"].str.contains(
            "CacheClip",
            case=False,
            na=False,
        )
    ].copy()

    if len(cacheclip_df) == 0:
        return

    token_type_order = [
        "entity_like",
        "digit",
        "underscore",
        "punctuation",
        "newline",
        "whitespace",
        "eos_like",
        "other",
    ]

    token_type_order = [
        token_type
        for token_type in token_type_order
        if token_type in cacheclip_df[
            "token_type"
        ].unique()
    ]

    if len(token_type_order) == 0:
        return

    fig, ax = plt.subplots(
        figsize=(11.5, 6.0),
    )

    sns.boxplot(
        data=cacheclip_df,
        x="token_type",
        y="kl",
        order=token_type_order,
        color="#B39DDB",
        showfliers=False,
        width=0.58,
        boxprops={
            "edgecolor": "#333333",
            "facecolor": "#D1C4E9",
        },
        medianprops={
            "color": "#111111",
            "linewidth": 1.6,
        },
        whiskerprops={
            "color": "#444444",
        },
        capprops={
            "color": "#444444",
        },
        ax=ax,
    )

    # 叠加均值点。
    means = (
        cacheclip_df.groupby(
            "token_type"
        )["kl"]
        .mean()
        .reindex(token_type_order)
    )

    ax.scatter(
        np.arange(len(token_type_order)),
        means.to_numpy(),
        color="#D95F02",
        marker="D",
        s=52,
        edgecolor="black",
        linewidth=0.65,
        label="Mean KL",
        zorder=5,
    )

    ax.set_xlabel("Reference Generated Token Type")
    ax.set_ylabel("Step-wise KL Divergence")

    ax.set_title(
        "KL Divergence by Reference Generated Token Type",
        pad=12,
        fontweight="bold",
    )

    ax.set_yscale("log")

    ax.tick_params(
        axis="x",
        rotation=25,
    )

    ax.legend(
        loc="upper right",
        frameon=True,
    )

    sns.despine(ax=ax)

    save_figure(
        fig,
        output_dir,
        "03_token_type_kl_distribution",
    )


# ============================================================
# 13. Figure 4：High-KL Token Frequency
# ============================================================

def plot_high_kl_token_distribution(
    df,
    output_dir,
    quantile=0.90,
):
    """
    选择每个方法内部 KL 的 top quantile 行，
    统计高 KL 位置对应的 reference token。

    该图帮助判断：
        高 KL 是否集中在某些具体 token。
    """

    cacheclip_df = df[
        df["method"].str.contains(
            "CacheClip",
            case=False,
            na=False,
        )
    ].copy()

    if len(cacheclip_df) == 0:
        return

    threshold = cacheclip_df["kl"].quantile(
        quantile
    )

    high_df = cacheclip_df[
        cacheclip_df["kl"] >= threshold
    ].copy()

    if len(high_df) == 0:
        return

    token_counts = (
        high_df.groupby(
            "reference_token_text_clean"
        )
        .size()
        .sort_values(
            ascending=False
        )
        .head(20)
    )

    if len(token_counts) == 0:
        return

    fig, ax = plt.subplots(
        figsize=(12.0, 6.5),
    )

    labels = token_counts.index.tolist()
    values = token_counts.to_numpy()

    ax.bar(
        np.arange(len(labels)),
        values,
        color="#7570B3",
        edgecolor="black",
        linewidth=0.65,
    )

    ax.set_xticks(
        np.arange(len(labels))
    )

    ax.set_xticklabels(
        labels,
        rotation=65,
        ha="right",
    )

    ax.set_xlabel(
        "Reference Token Text at Top-KL Positions"
    )

    ax.set_ylabel(
        "Occurrence Count"
    )

    ax.set_title(
        f"Reference Tokens at the Highest-{quantile:.0%} KL Positions",
        pad=12,
        fontweight="bold",
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    sns.despine(ax=ax)

    save_figure(
        fig,
        output_dir,
        "04_high_kl_token_distribution",
    )

    token_counts_df = token_counts.reset_index()
    token_counts_df.columns = [
        "reference_token_text",
        "count",
    ]

    token_counts_df.to_csv(
        output_dir
        / "04_high_kl_token_distribution.csv",
        index=False,
        encoding="utf-8",
    )


# ============================================================
# 14. Figure 5：Top-k / Cosine by Generation Step
# ============================================================

def plot_topk_cosine_by_step(
    step_df,
    output_dir,
):
    """
    左图：
        Top-k Overlap

    右图：
        Logits Cosine

    用于观察：
        KL 较高的位置是否同时有低 Top-k / 低 Cosine。
    """

    methods = sorted(
        step_df["method"].unique(),
        key=method_order,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16.0, 5.6),
        constrained_layout=True,
    )

    for method in methods:
        subset = step_df[
            step_df["method"] == method
        ].sort_values(
            "generation_step"
        )

        if len(subset) == 0:
            continue

        color = get_method_color(method)
        marker = get_method_marker(method)

        x = subset[
            "generation_step"
        ].to_numpy()

        topk = (
            subset[
                "mean_topk_overlap"
            ].to_numpy()
            * 100.0
        )

        cosine = subset[
            "mean_logits_cosine"
        ].to_numpy()

        axes[0].plot(
            x,
            topk,
            color=color,
            marker=marker,
            label=method,
        )

        axes[1].plot(
            x,
            cosine,
            color=color,
            marker=marker,
            label=method,
        )

    axes[0].set_title(
        "Top-k Overlap by Generation Step",
        fontweight="bold",
    )

    axes[0].set_xlabel("Generation Step")
    axes[0].set_ylabel("Top-k Overlap (%)")
    axes[0].set_ylim(0, 105)
    axes[0].yaxis.set_major_formatter(
        mtick.PercentFormatter()
    )

    axes[1].set_title(
        "Logits Cosine Similarity by Generation Step",
        fontweight="bold",
    )

    axes[1].set_xlabel("Generation Step")
    axes[1].set_ylabel("Logits Cosine Similarity")
    axes[1].set_ylim(0, 1.02)

    for ax in axes:
        ax.grid(
            axis="y",
            alpha=0.25,
        )

    handles, labels = axes[0].get_legend_handles_labels()

    axes[0].legend(
        handles,
        labels,
        loc="lower left",
        frameon=True,
        fontsize=8.2,
    )

    sns.despine()

    save_figure(
        fig,
        output_dir,
        "05_topk_cosine_by_generation_step",
    )


# ============================================================
# 15. 高风险 Step 汇总
# ============================================================

def save_high_risk_step_summary(
    step_df,
    output_dir,
    top_n=10,
):
    """
    保存各方法 KL 最高的 generation step。
    """

    rows = []

    for method in step_df["method"].unique():
        subset = step_df[
            step_df["method"] == method
        ].sort_values(
            "mean_kl",
            ascending=False,
        ).head(top_n)

        for _, row in subset.iterrows():
            rows.append(
                {
                    "method": method,
                    "repair_ratio": row[
                        "repair_ratio"
                    ],
                    "generation_step": row[
                        "generation_step"
                    ],
                    "n_samples": row[
                        "n_samples"
                    ],
                    "mean_kl": row[
                        "mean_kl"
                    ],
                    "ci_low_kl": row[
                        "ci_low_kl"
                    ],
                    "ci_high_kl": row[
                        "ci_high_kl"
                    ],
                    "mean_js": row[
                        "mean_js"
                    ],
                    "mean_logits_cosine": row[
                        "mean_logits_cosine"
                    ],
                    "mean_top1_match": row[
                        "mean_top1_match"
                    ],
                    "mean_topk_overlap": row[
                        "mean_topk_overlap"
                    ],
                }
            )

    high_risk_df = pd.DataFrame(rows)

    high_risk_path = (
        output_dir
        / "06_high_risk_generation_steps.csv"
    )

    high_risk_df.to_csv(
        high_risk_path,
        index=False,
        encoding="utf-8",
    )

    print(f"[Saved] {high_risk_path}")


# ============================================================
# 16. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect reference_token_text and generation-step "
            "KL divergence in CacheClip KV reuse."
        )
    )

    parser.add_argument(
        "--input_csv",
        type=str,
        default=(
            "./long_bench/"
            "cacheclip_generation_kl2/"
            "per_generation_step.csv"
        ),
        help=(
            "Path to per_generation_step.csv."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=(
            "./long_bench/"
            "cacheclip_generation_kl2/"
            "reference_token_analysis"
        ),
    )

    parser.add_argument(
        "--top_n",
        type=int,
        default=100,
        help=(
            "Number of highest-KL rows saved to "
            "01_top_kl_generation_tokens.csv."
        ),
    )

    parser.add_argument(
        "--bootstrap_samples",
        type=int,
        default=3000,
    )

    parser.add_argument(
        "--high_kl_quantile",
        type=float,
        default=0.90,
    )

    args = parser.parse_args()

    setup_paper_style()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not input_csv.exists():
        raise FileNotFoundError(
            f"Input CSV not found:\n{input_csv}"
        )

    if not (
        0.5
        <= args.high_kl_quantile
        < 1.0
    ):
        raise ValueError(
            "high_kl_quantile should be in [0.5, 1.0)."
        )

    print("=" * 110)
    print("Generation Token / KL Analysis")
    print("=" * 110)
    print(f"Input:  {input_csv}")
    print(f"Output: {output_dir}")
    print("=" * 110)

    df = load_generation_step_data(
        input_csv,
        eos_token_id=None,
    )

    print(
        f"Loaded rows: {len(df)}"
    )

    print(
        f"Methods: {df['method'].unique().tolist()}"
    )

    print(
        f"Samples: {df['sample_id'].nunique()}"
    )

    # --------------------------------------------------------
    # 1. 保存最高 KL 原始行
    # --------------------------------------------------------
    top_kl_df = extract_top_kl_rows(
        df,
        top_n=args.top_n,
    )

    top_kl_path = (
        output_dir
        / "01_top_kl_generation_tokens.csv"
    )

    top_kl_df.to_csv(
        top_kl_path,
        index=False,
        encoding="utf-8",
    )

    print(f"[Saved] {top_kl_path}")

    # --------------------------------------------------------
    # 2. 汇总
    # --------------------------------------------------------
    step_df = summarize_by_generation_step(
        df,
        n_bootstrap=args.bootstrap_samples,
    )

    step_summary_path = (
        output_dir
        / "03_generation_step_summary.csv"
    )

    step_df.to_csv(
        step_summary_path,
        index=False,
        encoding="utf-8",
    )

    print(f"[Saved] {step_summary_path}")

    token_summary_df = summarize_by_reference_token(
        df
    )

    token_summary_path = (
        output_dir
        / "02_reference_token_summary.csv"
    )

    token_summary_df.to_csv(
        token_summary_path,
        index=False,
        encoding="utf-8",
    )

    print(f"[Saved] {token_summary_path}")

    token_type_df = summarize_by_token_type(
        df
    )

    token_type_path = (
        output_dir
        / "04_reference_token_type_summary.csv"
    )

    token_type_df.to_csv(
        token_type_path,
        index=False,
        encoding="utf-8",
    )

    print(f"[Saved] {token_type_path}")

    # --------------------------------------------------------
    # 3. 可视化
    # --------------------------------------------------------
    plot_step_kl_curve(
        step_df,
        output_dir,
    )

    plot_step_kl_heatmap(
        step_df,
        output_dir,
    )

    plot_token_type_kl_distribution(
        df,
        output_dir,
    )

    plot_high_kl_token_distribution(
        df,
        output_dir,
        quantile=args.high_kl_quantile,
    )

    plot_topk_cosine_by_step(
        step_df,
        output_dir,
    )

    save_high_risk_step_summary(
        step_df,
        output_dir,
        top_n=10,
    )

    # --------------------------------------------------------
    # 4. 保存 metadata
    # --------------------------------------------------------
    metadata = {
        "input_csv": str(
            input_csv.resolve()
        ),
        "output_dir": str(
            output_dir.resolve()
        ),
        "top_n": args.top_n,
        "bootstrap_samples": (
            args.bootstrap_samples
        ),
        "high_kl_quantile": (
            args.high_kl_quantile
        ),
        "n_rows": int(len(df)),
        "n_samples": int(
            df["sample_id"].nunique()
        ),
        "methods": [
            str(x)
            for x in df["method"].unique()
        ],
        "token_type_definition": {
            "entity_like": (
                "Token text contains an entity-like pattern such as E_001."
            ),
            "digit": (
                "Token text is purely numeric."
            ),
            "underscore": (
                "Token contains underscore."
            ),
            "punctuation": (
                "Token consists of punctuation symbols."
            ),
            "newline": (
                "Token contains newline marker."
            ),
            "whitespace": (
                "Token is whitespace."
            ),
            "eos_like": (
                "Token ID equals EOS ID when EOS ID is supplied."
            ),
            "other": (
                "All other tokens."
            ),
        },
    }

    metadata_path = (
        output_dir
        / "07_analysis_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[Saved] {metadata_path}")

    # --------------------------------------------------------
    # 5. 打印 Top-KL Token
    # --------------------------------------------------------
    print("\n" + "=" * 110)
    print("Top KL Generation Rows")
    print("=" * 110)

    print(
        top_kl_df.to_string(
            index=False,
            max_colwidth=40,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\n" + "=" * 110)
    print("Generation-Step Summary")
    print("=" * 110)

    print(
        step_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nCompleted.")
    print(
        f"Analysis results saved to:\n"
        f"{output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
           
