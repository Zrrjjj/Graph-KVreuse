#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
benchmark_cacheclip_generation_kl.py

Full-Generation Trajectory KL Benchmark for CacheClip
======================================================

目标：
    测试完整生成过程中的逐步 KL，
    判断 CacheClip KV cache reuse 在哪些生成位置偏差最大。

比较方法：
    1. Full Long-Path KV Reuse
    2. CacheClip Token Repair 10%
    3. CacheClip Token Repair 20%
    4. CacheClip Token Repair 30%
    5. CacheClip Token Repair 40%

参考：
    Full Recompute Target Prompt

核心评估方式：
    Teacher-forced generation trajectory

流程：
    1. Full Recompute 在 Target Prompt 上生成参考 token 序列；
    2. 每个 KV reuse 方法构造自己的初始 cache；
    3. 在每个 generation step：
         - 比较 Full 和 Method 当前 next-token distribution；
         - 记录逐步 KL / JS / Top-k；
         - 将 Full Recompute 生成的 token 输入两个 cache；
         - 进入下一步；
    4. 这样所有方法在同一生成前缀上比较。

为什么不直接比较自由生成轨迹：
    如果某个方法第 1 步生成了不同 token，
    第 2 步的输入上下文就不同，
    后续 KL 同时混入：
        - cache approximation error
        - generated-prefix divergence
    不利于定位 KV reuse 的具体误差位置。

重要：
    - Full Recompute 的 logits 作为参考；
    - Full Recompute 先产生参考答案 token 序列；
    - CacheClip selector 使用 auxiliary small model；
    - Patch 使用 Target Full KV，因此属于 Oracle KV Patch；
    - 当前 KL 是生成轨迹级别的 teacher-forced KL；
    - 不是线上可部署 latency benchmark。
"""

import sys
import gc
import json
import math
import time
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns


# ============================================================
# 1. 导入原始 CacheClip token benchmark
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent

ORIGINAL_SCRIPT = (
    CURRENT_DIR / "benchmark_longpath_cacheclip_token.py"
)

if not ORIGINAL_SCRIPT.exists():
    raise FileNotFoundError(
        "\nCannot find:\n"
        f"{ORIGINAL_SCRIPT}\n\n"
        "Please place benchmark_cacheclip_generation_kl.py "
        "and benchmark_longpath_cacheclip_token.py "
        "in the same directory."
    )


spec = importlib.util.spec_from_file_location(
    "original_cacheclip_token_benchmark",
    str(ORIGINAL_SCRIPT),
)

original = importlib.util.module_from_spec(spec)
sys.modules["original_cacheclip_token_benchmark"] = original
spec.loader.exec_module(original)


# 使用原始脚本中的函数。
parse_dtype = original.parse_dtype
get_model_device = original.get_model_device
measure_time = original.measure_time
to_legacy_cache = original.to_legacy_cache
extract_prompt_spans = original.extract_prompt_spans
load_model = original.load_model
full_prefill = original.full_prefill
calculate_small_token_attention_scores = (
    original.calculate_small_attention_scores
)
map_small_scores_to_large_tokens = (
    original.map_small_scores_to_large_tokens
)
select_top_ratio_tokens = original.select_top_ratio_tokens
build_hybrid_path_cache = original.build_hybrid_path_cache
recompute_target_suffix = original.recompute_target_suffix
greedy_generate = original.greedy_generate
build_candidates = original.build_candidates
rank_candidates = original.rank_candidates


# ============================================================
# 2. 论文风格
# ============================================================

def setup_paper_style():
    sns.set_theme(
        style="whitegrid",
        context="paper",
    )

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 400,
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
        dpi=400,
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
# 3. Span / Token 对齐辅助函数
# ============================================================

def find_question_char_span(prompt):
    question_marker = "Question: "
    answer_marker = "\nAnswer:"

    question_start = prompt.rfind(question_marker)

    if question_start < 0:
        raise RuntimeError(
            "Cannot find Question marker."
        )

    question_start += len(question_marker)

    question_end = prompt.find(
        answer_marker,
        question_start,
    )

    if question_end < 0:
        raise RuntimeError(
            "Cannot find Answer marker."
        )

    return question_start, question_end


def token_positions_to_contiguous_span(positions):
    if len(positions) == 0:
        raise RuntimeError(
            "Empty token position list."
        )

    return (
        positions[0],
        positions[-1] + 1,
    )


def verify_source_target_path_alignment(
    source_spans,
    target_spans,
):
    source_ids = source_spans["input_ids"]
    target_ids = target_spans["input_ids"]

    source_evidence = source_spans[
        "evidence_positions"
    ]

    target_evidence = target_spans[
        "evidence_positions"
    ]

    if len(source_evidence) != len(target_evidence):
        raise RuntimeError(
            "Source/Target evidence token count differs."
        )

    source_evidence_ids = source_ids[
        source_evidence
    ]

    target_evidence_ids = target_ids[
        target_evidence
    ]

    if not torch.equal(
        source_evidence_ids,
        target_evidence_ids,
    ):
        raise RuntimeError(
            "Source/Target evidence token IDs differ."
        )

    source_path_start, source_path_end = (
        source_spans["full_path_token_span"]
    )

    target_path_start, target_path_end = (
        target_spans["full_path_token_span"]
    )

    source_path_ids = source_ids[
        source_path_start:source_path_end
    ]

    target_path_ids = target_ids[
        target_path_start:target_path_end
    ]

    if not torch.equal(
        source_path_ids,
        target_path_ids,
    ):
        raise RuntimeError(
            "Source/Target full target-path token IDs differ."
        )


# ============================================================
# 4. CacheClip Token Selector
# ============================================================

def calculate_cacheclip_token_scores(
    small_model,
    small_tokenizer,
    small_device,
    target_prompt,
    target_large_spans,
    hop_length,
    small_attention_last_k,
):
    """
    用小模型计算：

        Question -> Target Evidence Token Attention

    再映射到大模型的 Target Evidence Token 空间。
    """

    target_small_spans = extract_prompt_spans(
        small_tokenizer,
        target_prompt,
        hop_length,
    )

    small_result = full_prefill(
        model=small_model,
        input_ids_cpu=target_small_spans["input_ids"],
        device=small_device,
        output_attentions=True,
    )

    attention_result = (
        calculate_small_token_attention_scores(
            attentions=small_result["attentions"],
            question_positions=target_small_spans[
                "question_positions"
            ],
            evidence_positions=target_small_spans[
                "evidence_positions"
            ],
            last_k=small_attention_last_k,
        )
    )

    mapped_scores = map_small_scores_to_large_tokens(
        small_scores=attention_result["final_scores"],
        small_offsets=target_small_spans["offsets"],
        small_evidence_positions=target_small_spans[
            "evidence_positions"
        ],
        large_offsets=target_large_spans["offsets"],
        large_evidence_positions=target_large_spans[
            "evidence_positions"
        ],
    )

    selected_layers = attention_result[
        "selected_layers"
    ]

    del small_result

    return mapped_scores, selected_layers


# ============================================================
# 5. Cache 初始化
# ============================================================

def build_method_initial_state(
    method_name,
    repair_ratio,
    source_full,
    target_full,
    source_spans,
    target_spans,
    target_evidence_positions,
    cacheclip_scores,
    model,
    device,
):
    """
    构造某个方法在 Target Prompt 末尾的初始状态。

    返回：
        logits
        cache
        seq_len
        patch metadata
    """

    source_path_span = source_spans[
        "full_path_token_span"
    ]

    target_path_span = target_spans[
        "full_path_token_span"
    ]

    if method_name == "Full Recompute":
        return {
            "logits": target_full["logits"],
            "cache": target_full["cache"],
            "seq_len": target_full["seq_len"],
            "selected_indices": list(
                range(len(target_evidence_positions))
            ),
            "selected_ratio": 1.0,
        }

    if method_name == "Full Long-Path KV Reuse":
        selected_indices = []

    else:
        selected_indices, _ = select_top_ratio_tokens(
            scores=cacheclip_scores,
            ratio=repair_ratio,
        )

    hybrid_cache = build_hybrid_path_cache(
        source_cache=source_full["cache"],
        target_cache=target_full["cache"],
        source_full_path_span=source_spans[
            "full_path_token_span"
        ],
        target_full_path_span=target_spans[
            "full_path_token_span"
        ],
        target_evidence_positions=target_evidence_positions,
        selected_local_indices=selected_indices,
    )

    method_result = recompute_target_suffix(
        model=model,
        hybrid_cache=hybrid_cache,
        target_input_ids_cpu=target_spans["input_ids"],
        target_full_path_span=target_spans[
            "full_path_token_span"
        ],
        device=device,
    )

    return {
        "logits": method_result["logits"],
        "cache": method_result["cache"],
        "seq_len": method_result["seq_len"],
        "selected_indices": selected_indices,
        "selected_ratio": (
            len(selected_indices)
            / max(len(target_evidence_positions), 1)
        ),
    }


# ============================================================
# 6. 逐步 Teacher-Forced Forward
# ============================================================

@torch.no_grad()
def advance_one_token(
    model,
    token_id,
    cache,
    current_seq_len,
    device,
):
    """
    将一个参考生成 token 输入当前 cache，
    得到下一个位置的 logits 和更新后的 cache。

    参数：
        token_id:
            当前要输入的 token id。

        cache:
            当前 past_key_values。

        current_seq_len:
            输入 token 之前已有的序列长度。

    返回：
        next_logits
        next_cache
        next_seq_len
    """

    input_token = torch.tensor(
        [[int(token_id)]],
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.ones(
        (1, current_seq_len + 1),
        dtype=torch.long,
        device=device,
    )

    position_ids = torch.tensor(
        [[current_seq_len]],
        dtype=torch.long,
        device=device,
    )

    outputs = model(
        input_ids=input_token,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
    )

    return {
        "logits": outputs.logits[:, -1, :].detach(),
        "cache": to_legacy_cache(
            outputs.past_key_values
        ),
        "seq_len": current_seq_len + 1,
    }


# ============================================================
# 7. KL / JS / Top-k 指标
# ============================================================

def calculate_step_distribution_metrics(
    full_logits,
    method_logits,
    top_k,
):
    """
    对一个 generation step 计算：

        KL(P_full || P_method)
        JS(P_full, P_method)
        logits cosine
        top1 match
        top-k overlap
        full top1 token
        method top1 token
    """

    full_logits = full_logits.float()
    method_logits = method_logits.float()

    full_log_probs = F.log_softmax(
        full_logits,
        dim=-1,
    )

    method_log_probs = F.log_softmax(
        method_logits,
        dim=-1,
    )

    full_probs = full_log_probs.exp()
    method_probs = method_log_probs.exp()

    kl = F.kl_div(
        method_log_probs,
        full_probs,
        reduction="batchmean",
    ).item()

    # 数值误差导致的极小负数截断为 0。
    if abs(kl) < 1e-8:
        kl = 0.0

    mixture_probs = (
        0.5 * full_probs
        + 0.5 * method_probs
    )

    mixture_log_probs = torch.log(
        mixture_probs + 1e-12
    )

    js = 0.5 * (
        F.kl_div(
            mixture_log_probs,
            full_probs,
            reduction="batchmean",
        )
        + F.kl_div(
            mixture_log_probs,
            method_probs,
            reduction="batchmean",
        )
    ).item()

    if abs(js) < 1e-8:
        js = 0.0

    cosine = F.cosine_similarity(
        full_logits,
        method_logits,
        dim=-1,
    ).mean().item()

    full_top1 = int(
        torch.argmax(
            full_logits,
            dim=-1,
        ).item()
    )

    method_top1 = int(
        torch.argmax(
            method_logits,
            dim=-1,
        ).item()
    )

    full_topk = set(
        torch.topk(
            full_logits,
            k=top_k,
            dim=-1,
        ).indices[0].tolist()
    )

    method_topk = set(
        torch.topk(
            method_logits,
            k=top_k,
            dim=-1,
        ).indices[0].tolist()
    )

    full_top1_prob = float(
        full_probs[0, full_top1].item()
    )

    method_full_top1_prob = float(
        method_probs[0, full_top1].item()
    )

    return {
        "kl": kl,
        "js": js,
        "logits_cosine": cosine,
        "top1_match": int(
            full_top1 == method_top1
        ),
        "topk_overlap": len(
            full_topk & method_topk
        ) / float(top_k),
        "full_top1_token_id": full_top1,
        "method_top1_token_id": method_top1,
        "full_top1_probability": full_top1_prob,
        "method_probability_on_full_top1": (
            method_full_top1_prob
        ),
    }


# ============================================================
# 8. 单样本完整生成轨迹 KL
# ============================================================

@torch.no_grad()
def evaluate_generation_trajectory(
    model,
    tokenizer,
    full_initial_state,
    method_initial_state,
    reference_token_ids,
    device,
    sample_id,
    method_name,
    repair_ratio,
    top_k,
):
    """
    Teacher-forced generation trajectory evaluation。

    重要：
        reference_token_ids 是 Full Recompute 的 greedy 输出。

    每一步：
        1. 当前 Full 和 Method cache 都看到相同的历史 token；
        2. 比较二者对下一个 token 的 logits；
        3. 将 Full 生成的 reference token 同时输入二者；
        4. 进入下一步。

    这样测到的是：
        在相同生成历史条件下，
        Method 与 Full Recompute 的逐步分布差异。
    """

    full_state = {
        "logits": full_initial_state["logits"],
        "cache": full_initial_state["cache"],
        "seq_len": full_initial_state["seq_len"],
    }

    method_state = {
        "logits": method_initial_state["logits"],
        "cache": method_initial_state["cache"],
        "seq_len": method_initial_state["seq_len"],
    }

    rows = []

    for generation_step, token_id in enumerate(
        reference_token_ids,
        start=1,
    ):
        metrics = calculate_step_distribution_metrics(
            full_logits=full_state["logits"],
            method_logits=method_state["logits"],
            top_k=top_k,
        )

        token_text = tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=False,
        )

        row = {
            "sample_id": sample_id,
            "method": method_name,
            "repair_ratio": repair_ratio,
            "generation_step": generation_step,
            "history_length_before_token": (
                full_state["seq_len"]
            ),
            "reference_token_id": int(token_id),
            "reference_token_text": token_text,
            **metrics,
        }

        rows.append(row)

        # 将同一个 Full reference token 输入 Full 和 Method。
        full_state = advance_one_token(
            model=model,
            token_id=token_id,
            cache=full_state["cache"],
            current_seq_len=full_state["seq_len"],
            device=device,
        )

        method_state = advance_one_token(
            model=model,
            token_id=token_id,
            cache=method_state["cache"],
            current_seq_len=method_state["seq_len"],
            device=device,
        )

    return rows


# ============================================================
# 9. 生成参考 token 序列
# ============================================================

def get_reference_generation_ids(
    model,
    tokenizer,
    target_full,
    device,
    max_new_tokens,
):
    """
    用 Full Recompute 的 prompt 状态生成 greedy reference token 序列。

    这里不保存完整 decode cache，
    只保存生成出的 token id。
    """

    generated_text = greedy_generate(
        model=model,
        tokenizer=tokenizer,
        initial_logits=target_full["logits"],
        initial_cache=target_full["cache"],
        initial_seq_len=target_full["seq_len"],
        device=device,
        max_new_tokens=max_new_tokens,
    )

    reference_ids = tokenizer.encode(
        generated_text,
        add_special_tokens=False,
    )

    return reference_ids, generated_text


# ============================================================
# 10. 单样本主流程
# ============================================================

def run_one_sample(
    record,
    small_tokenizer,
    small_model,
    small_device,
    large_tokenizer,
    large_model,
    large_device,
    args,
):
    sample_id = record["sample_id"]
    hop_length = int(record["hop_length"])

    source_prompt = record["source_prompt"]
    target_prompt = record["target_prompt"]

    # --------------------------------------------------------
    # Span extraction
    # --------------------------------------------------------
    source_large_spans = extract_prompt_spans(
        large_tokenizer,
        source_prompt,
        hop_length,
    )

    target_large_spans = extract_prompt_spans(
        large_tokenizer,
        target_prompt,
        hop_length,
    )

    target_small_spans = extract_prompt_spans(
        small_tokenizer,
        target_prompt,
        hop_length,
    )

    source_evidence_positions = (
        source_large_spans["evidence_positions"]
    )

    target_evidence_positions = (
        target_large_spans["evidence_positions"]
    )

    if len(source_evidence_positions) != len(
        target_evidence_positions
    ):
        raise RuntimeError(
            "Source/Target evidence token count mismatch."
        )

    source_evidence_ids = source_large_spans[
        "input_ids"
    ][source_evidence_positions]

    target_evidence_ids = target_large_spans[
        "input_ids"
    ][target_evidence_positions]

    if not torch.equal(
        source_evidence_ids,
        target_evidence_ids,
    ):
        raise RuntimeError(
            "Source/Target evidence token IDs differ."
        )

    source_full_path_span = source_large_spans[
        "full_path_token_span"
    ]

    target_full_path_span = target_large_spans[
        "full_path_token_span"
    ]

    source_path_start, source_path_end = (
        source_full_path_span
    )

    target_path_start, target_path_end = (
        target_full_path_span
    )

    source_path_ids = source_large_spans[
        "input_ids"
    ][source_path_start:source_path_end]

    target_path_ids = target_large_spans[
        "input_ids"
    ][target_path_start:target_path_end]

    if not torch.equal(
        source_path_ids,
        target_path_ids,
    ):
        raise RuntimeError(
            "Source/Target full KG_PATH token IDs differ."
        )

    # --------------------------------------------------------
    # 1. Small model CacheClip selector
    # --------------------------------------------------------
    small_result, small_selector_latency = measure_time(
        lambda: full_prefill(
            model=small_model,
            input_ids_cpu=target_small_spans[
                "input_ids"
            ],
            device=small_device,
            output_attentions=True,
        ),
        small_device,
    )

    small_attention_result = (
        calculate_small_token_attention_scores(
            attentions=small_result["attentions"],
            question_positions=target_small_spans[
                "question_positions"
            ],
            evidence_positions=target_small_spans[
                "evidence_positions"
            ],
            last_k=args.small_attention_last_k,
        )
    )

    cacheclip_scores = (
        map_small_scores_to_large_tokens(
            small_scores=small_attention_result[
                "final_scores"
            ],
            small_offsets=target_small_spans[
                "offsets"
            ],
            small_evidence_positions=target_small_spans[
                "evidence_positions"
            ],
            large_offsets=target_large_spans[
                "offsets"
            ],
            large_evidence_positions=target_large_spans[
                "evidence_positions"
            ],
        )
    )

    # --------------------------------------------------------
    # 2. 大模型 Source / Target full prefill
    # --------------------------------------------------------
    source_full, source_prefill_latency = measure_time(
        lambda: full_prefill(
            model=large_model,
            input_ids_cpu=source_large_spans[
                "input_ids"
            ],
            device=large_device,
            output_attentions=False,
        ),
        large_device,
    )

    target_full, target_prefill_latency = measure_time(
        lambda: full_prefill(
            model=large_model,
            input_ids_cpu=target_large_spans[
                "input_ids"
            ],
            device=large_device,
            output_attentions=False,
        ),
        large_device,
    )

    # --------------------------------------------------------
    # 3. Full Recompute 的参考生成轨迹
    # --------------------------------------------------------
    reference_token_ids, full_generated_text = (
        get_reference_generation_ids(
            model=large_model,
            tokenizer=large_tokenizer,
            target_full=target_full,
            device=large_device,
            max_new_tokens=args.max_new_tokens,
        )
    )

    # --------------------------------------------------------
    # 4. 构造 Full Recompute state
    # --------------------------------------------------------
    full_state = {
        "logits": target_full["logits"],
        "cache": target_full["cache"],
        "seq_len": target_full["seq_len"],
    }

    # --------------------------------------------------------
    # 5. 构造 Full KV Reuse state
    # --------------------------------------------------------
    reuse_state, reuse_construction_latency = measure_time(
        lambda: build_method_initial_state(
            method_name="Full Long-Path KV Reuse",
            repair_ratio=0.0,
            source_full=source_full,
            target_full=target_full,
            source_spans=source_large_spans,
            target_spans=target_large_spans,
            target_evidence_positions=target_evidence_positions,
            cacheclip_scores=cacheclip_scores,
            model=large_model,
            device=large_device,
        ),
        large_device,
    )

    # --------------------------------------------------------
    # 6. 先跑 Full KV Reuse trajectory
    # --------------------------------------------------------
    trajectory_rows = []

    reuse_trajectory_rows = (
        evaluate_generation_trajectory(
            model=large_model,
            tokenizer=large_tokenizer,
            full_initial_state=full_state,
            method_initial_state=reuse_state,
            reference_token_ids=reference_token_ids,
            device=large_device,
            sample_id=sample_id,
            method_name="Full Long-Path KV Reuse",
            repair_ratio=0.0,
            top_k=args.top_k,
        )
    )

    trajectory_rows.extend(
        reuse_trajectory_rows
    )

    # --------------------------------------------------------
    # 7. CacheClip 各 repair ratio
    # --------------------------------------------------------
    metadata_rows = []

    for repair_ratio in args.repair_ratios:
        selected_indices, requested_budget = (
            select_top_ratio_tokens(
                scores=cacheclip_scores,
                ratio=repair_ratio,
            )
        )

        actual_ratio = (
            len(selected_indices)
            / float(
                max(
                    len(target_evidence_positions),
                    1,
                )
            )
        )

        cacheclip_state, cacheclip_construction_latency = (
            measure_time(
                lambda: build_method_initial_state(
                    method_name="CacheClip",
                    repair_ratio=repair_ratio,
                    source_full=source_full,
                    target_full=target_full,
                    source_spans=source_large_spans,
                    target_spans=target_large_spans,
                    target_evidence_positions=(
                        target_evidence_positions
                    ),
                    cacheclip_scores=cacheclip_scores,
                    model=large_model,
                    device=large_device,
                ),
                large_device,
            )
        )

        method_name = (
            f"CacheClip Token Repair "
            f"{repair_ratio:.0%}"
        )

        cacheclip_trajectory_rows = (
            evaluate_generation_trajectory(
                model=large_model,
                tokenizer=large_tokenizer,
                full_initial_state=full_state,
                method_initial_state=cacheclip_state,
                reference_token_ids=reference_token_ids,
                device=large_device,
                sample_id=sample_id,
                method_name=method_name,
                repair_ratio=repair_ratio,
                top_k=args.top_k,
            )
        )

        # 为每个 generation step 添加实际 repair 信息。
        for row in cacheclip_trajectory_rows:
            row["requested_repair_ratio"] = repair_ratio
            row["actual_repair_token_ratio"] = actual_ratio
            row["selected_token_count"] = len(selected_indices)
            row["requested_token_budget"] = requested_budget

        trajectory_rows.extend(
            cacheclip_trajectory_rows
        )

        metadata_rows.append(
            {
                "sample_id": sample_id,
                "method": method_name,
                "repair_ratio": repair_ratio,
                "requested_token_budget": requested_budget,
                "selected_token_count": len(selected_indices),
                "actual_repair_token_ratio": actual_ratio,
                "evidence_token_count": len(
                    target_evidence_positions
                ),
                "selected_local_indices": json.dumps(
                    selected_indices
                ),
                "attention_layers": json.dumps(
                    small_attention_result[
                        "selected_layers"
                    ]
                ),
                "small_selector_latency_sec": (
                    small_selector_latency
                ),
                "source_prefill_latency_sec": (
                    source_prefill_latency
                ),
                "target_prefill_latency_sec": (
                    target_prefill_latency
                ),
                "cacheclip_construction_latency_sec": (
                    cacheclip_construction_latency
                ),
                "reference_generated_text": full_generated_text,
            }
        )

        del cacheclip_state

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 保存当前样本的参考生成结果。
    metadata_rows.append(
        {
            "sample_id": sample_id,
            "method": "Full Recompute",
            "repair_ratio": 1.0,
            "requested_token_budget": len(
                target_evidence_positions
            ),
            "selected_token_count": len(
                target_evidence_positions
            ),
            "actual_repair_token_ratio": 1.0,
            "evidence_token_count": len(
                target_evidence_positions
            ),
            "selected_local_indices": json.dumps(
                list(
                    range(
                        len(target_evidence_positions)
                    )
                )
            ),
            "attention_layers": json.dumps(
                small_attention_result[
                    "selected_layers"
                ]
            ),
            "small_selector_latency_sec": (
                small_selector_latency
            ),
            "source_prefill_latency_sec": 0.0,
            "target_prefill_latency_sec": (
                target_prefill_latency
            ),
            "cacheclip_construction_latency_sec": 0.0,
            "reference_generated_text": full_generated_text,
        }
    )

    # 保存 Direct / CacheClip 的局部选择结果。
    # Full Recompute 自身不需要作为 method state 运行 trajectory，
    # 因为 Full 是参考轨迹。
    del reuse_state

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        trajectory_rows,
        metadata_rows,
    )


# ============================================================
# 12. 逐步聚合
# ============================================================

def aggregate_by_generation_step(
    trajectory_df,
):
    """
    按 method / repair ratio / generation step 聚合。

    其中：
        step_mean_kl
        step_std_kl
        step_mean_js
        step_mean_topk_overlap
        step_mean_logits_cosine
    """

    grouped = (
        trajectory_df.groupby(
            [
                "method",
                "repair_ratio",
                "generation_step",
            ],
            as_index=False,
        )
        .agg(
            n_samples=(
                "sample_id",
                "nunique",
            ),
            step_mean_kl=(
                "kl",
                "mean",
            ),
            step_std_kl=(
                "kl",
                "std",
            ),
            step_mean_js=(
                "js",
                "mean",
            ),
            step_std_js=(
                "js",
                "std",
            ),
            step_mean_logits_cosine=(
                "logits_cosine",
                "mean",
            ),
            step_mean_top1_match=(
                "top1_match",
                "mean",
            ),
            step_mean_topk_overlap=(
                "topk_overlap",
                "mean",
            ),
            step_mean_full_top1_probability=(
                "full_top1_probability",
                "mean",
            ),
            step_mean_method_probability_on_full_top1=(
                "method_probability_on_full_top1",
                "mean",
            ),
        )
    )

    return grouped


def aggregate_by_method(
    trajectory_df,
):
    """
    聚合完整生成轨迹的整体指标。

    这里的 mean_kl 是：
        先对每个 sample 的所有 generation step 求平均，
        再对 sample 求平均。

    这样每个 sample 权重相同。
    """

    per_sample = (
        trajectory_df.groupby(
            [
                "sample_id",
                "method",
                "repair_ratio",
            ],
            as_index=False,
        )
        .agg(
            sample_mean_kl=(
                "kl",
                "mean",
            ),
            sample_mean_js=(
                "js",
                "mean",
            ),
            sample_mean_logits_cosine=(
                "logits_cosine",
                "mean",
            ),
            sample_top1_match=(
                "top1_match",
                "mean",
            ),
            sample_mean_topk_overlap=(
                "topk_overlap",
                "mean",
            ),
            sample_generation_steps=(
                "generation_step",
                "count",
            ),
        )
    )

    rows = []

    for (
        method,
        repair_ratio,
    ), group in per_sample.groupby(
        [
            "method",
            "repair_ratio",
        ]
    ):
        row = {
            "method": method,
            "repair_ratio": repair_ratio,
            "n_samples": group["sample_id"].nunique(),
        }

        for column in [
            "sample_mean_kl",
            "sample_mean_js",
            "sample_mean_logits_cosine",
            "sample_top1_match",
            "sample_mean_topk_overlap",
            "sample_generation_steps",
        ]:
            values = group[column].to_numpy(dtype=float)

            row[f"mean_{column}"] = float(
                np.nanmean(values)
            )

            row[f"std_{column}"] = float(
                np.nanstd(values)
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 13. 画图
# ============================================================

METHOD_STYLE = {
    "Full Long-Path KV Reuse": {
        "label": "Full Long-Path KV Reuse",
        "color": "#D95F02",
        "marker": "D",
    },
    "CacheClip Token Repair 10%": {
        "label": "CacheClip 10%",
        "color": "#8DA0CB",
        "marker": "o",
    },
    "CacheClip Token Repair 20%": {
        "label": "CacheClip 20%",
        "color": "#7570B3",
        "marker": "s",
    },
    "CacheClip Token Repair 30%": {
        "label": "CacheClip 30%",
        "color": "#4C78A8",
        "marker": "^",
    },
    "CacheClip Token Repair 40%": {
        "label": "CacheClip 40%",
        "color": "#1B9E77",
        "marker": "P",
    },
}


def method_style(method_name):
    if method_name in METHOD_STYLE:
        return METHOD_STYLE[method_name]

    if "CacheClip" in method_name:
        ratio = method_name.split()[-1]

        return {
            "label": method_name,
            "color": "#7570B3",
            "marker": "o",
        }

    return {
        "label": method_name,
        "color": "#555555",
        "marker": "o",
    }


def plot_generation_kl_curve(
    step_df,
    output_dir,
):
    """
    Figure 1：
        generation step vs KL

    不同 repair ratio 分别画线。
    """

    fig, ax = plt.subplots(
        figsize=(10.5, 6.2),
    )

    methods = [
        "Full Long-Path KV Reuse",
        "CacheClip Token Repair 10%",
        "CacheClip Token Repair 20%",
        "CacheClip Token Repair 30%",
        "CacheClip Token Repair 40%",
    ]

    for method in methods:
        subset = step_df[
            step_df["method"] == method
        ].sort_values("generation_step")

        if len(subset) == 0:
            continue

        style = method_style(method)

        x = subset["generation_step"].to_numpy()
        y = subset["step_mean_kl"].to_numpy()
        std = subset["step_std_kl"].fillna(0).to_numpy()

        ax.plot(
            x,
            y,
            color=style["color"],
            marker=style["marker"],
            label=style["label"],
            zorder=4,
        )

        ax.fill_between(
            x,
            np.maximum(0, y - std),
            y + std,
            color=style["color"],
            alpha=0.12,
            linewidth=0,
            zorder=1,
        )

    ax.set_xlabel(
        "Generation Step along Full-Recompute Reference Trajectory"
    )

    ax.set_ylabel(
        "KL Divergence "
        r"$D_{\mathrm{KL}}(P_{\mathrm{Full}}\parallel P_{\mathrm{Method}})$"
    )

    ax.set_title(
        "Step-wise Distribution Divergence during Generation",
        pad=12,
        fontweight="bold",
    )

    ax.set_yscale("log")

    ax.grid(
        axis="both",
        alpha=0.25,
    )

    ax.legend(
        loc="best",
        frameon=True,
        fontsize=9,
    )

    sns.despine(ax=ax)

    save_figure(
        fig,
        output_dir,
        "01_generation_kl_curve",
    )


def plot_generation_js_curve(
    step_df,
    output_dir,
):
    """
    Figure 2：
        generation step vs JS divergence
    """

    fig, ax = plt.subplots(
        figsize=(10.5, 6.2),
    )

    methods = [
        "Full Long-Path KV Reuse",
        "CacheClip Token Repair 10%",
        "CacheClip Token Repair 20%",
        "CacheClip Token Repair 30%",
        "CacheClip Token Repair 40%",
    ]

    for method in methods:
        subset = step_df[
            step_df["method"] == method
        ].sort_values("generation_step")

        if len(subset) == 0:
            continue

        style = method_style(method)

        x = subset["generation_step"].to_numpy()
        y = subset["step_mean_js"].to_numpy()
        std = subset["step_std_js"].fillna(0).to_numpy()

        ax.plot(
            x,
            y,
            color=style["color"],
            marker=style["marker"],
            label=style["label"],
        )

        ax.fill_between(
            x,
            np.maximum(0, y - std),
            y + std,
            color=style["color"],
            alpha=0.12,
            linewidth=0,
        )

    ax.set_xlabel(
        "Generation Step along Full-Recompute Reference Trajectory"
    )

    ax.set_ylabel("Jensen–Shannon Divergence")

    ax.set_title(
        "Step-wise JS Divergence during Generation",
        pad=12,
        fontweight="bold",
    )

    ax.set_yscale("log")

    ax.legend(
        loc="best",
        frameon=True,
        fontsize=9,
    )

    sns.despine(ax=ax)

    save_figure(
        fig,
        output_dir,
        "02_generation_js_curve",
    )


def plot_generation_topk_curve(
    step_df,
    output_dir,
):
    """
    Figure 3：
        generation step vs Top-k Overlap
    """

    fig, ax = plt.subplots(
        figsize=(10.5, 6.2),
    )

    methods = [
        "Full Long-Path KV Reuse",
        "CacheClip Token Repair 10%",
        "CacheClip Token Repair 20%",
        "CacheClip Token Repair 30%",
        "CacheClip Token Repair 40%",
    ]

    for method in methods:
        subset = step_df[
            step_df["method"] == method
        ].sort_values("generation_step")

        if len(subset) == 0:
            continue

        style = method_style(method)

        x = subset["generation_step"].to_numpy()
        y = (
            subset["step_mean_topk_overlap"].to_numpy()
            * 100.0
        )

        ax.plot(
            x,
            y,
            color=style["color"],
            marker=style["marker"],
            label=style["label"],
        )

    ax.set_xlabel(
        "Generation Step along Full-Recompute Reference Trajectory"
    )

    ax.set_ylabel("Top-k Overlap (%)")

    ax.set_title(
        "Step-wise Next-token Top-k Agreement",
        pad=12,
        fontweight="bold",
    )

    ax.set_ylim(0, 105)

    ax.yaxis.set_major_formatter(
        mtick.PercentFormatter()
    )

    ax.legend(
        loc="lower right",
        frameon=True,
        fontsize=9,
    )

    sns.despine(ax=ax)

    save_figure(
        fig,
        output_dir,
        "03_generation_topk_curve",
    )


def plot_step_heatmap(
    step_df,
    output_dir,
):
    """
    Figure 4：
        method × generation step KL heatmap
    """

    methods = [
        "Full Long-Path KV Reuse",
        "CacheClip Token Repair 10%",
        "CacheClip Token Repair 20%",
        "CacheClip Token Repair 30%",
        "CacheClip Token Repair 40%",
    ]

    data = step_df[
        step_df["method"].isin(methods)
    ].copy()

    if len(data) == 0:
        return

    matrix = data.pivot(
        index="method",
        columns="generation_step",
        values="step_mean_kl",
    )

    matrix = matrix.reindex(
        methods
    )

    matrix.index = [
        method_style(method)["label"]
        for method in matrix.index
    ]

    fig, ax = plt.subplots(
        figsize=(13.5, 5.3),
    )

    sns.heatmap(
        matrix,
        cmap="magma",
        linewidths=0.45,
        linecolor="white",
        annot=True,
        fmt=".1e",
        cbar_kws={
            "label": "Step-wise KL"
        },
        ax=ax,
    )

    ax.set_title(
        "Generation-step KL Divergence Heatmap",
        pad=12,
        fontweight="bold",
    )

    ax.set_xlabel("Generation Step")
    ax.set_ylabel("Method")

    save_figure(
        fig,
        output_dir,
        "04_generation_step_kl_heatmap",
    )


def plot_step_position_summary(
    step_df,
    output_dir,
):
    """
    Figure 5：
        generation step 上的平均 cosine / top-k 双轴图
    """

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15.5, 5.5),
        constrained_layout=True,
    )

    methods = [
        "Full Long-Path KV Reuse",
        "CacheClip Token Repair 10%",
        "CacheClip Token Repair 20%",
        "CacheClip Token Repair 30%",
        "CacheClip Token Repair 40%",
    ]

    for method in methods:
        subset = step_df[
            step_df["method"] == method
        ].sort_values("generation_step")

        if len(subset) == 0:
            continue

        style = method_style(method)

        x = subset["generation_step"].to_numpy()

        cosine = subset[
            "step_mean_logits_cosine"
        ].to_numpy()

        topk = subset[
            "step_mean_topk_overlap"
        ].to_numpy() * 100.0

        axes[0].plot(
            x,
            cosine,
            color=style["color"],
            marker=style["marker"],
            label=style["label"],
        )

        axes[1].plot(
            x,
            topk,
            color=style["color"],
            marker=style["marker"],
            label=style["label"],
        )

    axes[0].set_title(
        "Logits Cosine Similarity",
        fontweight="bold",
    )

    axes[0].set_xlabel("Generation Step")
    axes[0].set_ylabel("Cosine Similarity")
    axes[0].set_ylim(0.8, 1.01)

    axes[1].set_title(
        "Top-k Overlap",
        fontweight="bold",
    )

    axes[1].set_xlabel("Generation Step")
    axes[1].set_ylabel("Top-k Overlap (%)")
    axes[1].set_ylim(0, 105)
    axes[1].yaxis.set_major_formatter(
        mtick.PercentFormatter()
    )

    for ax in axes:
        ax.grid(
            axis="y",
            alpha=0.25,
        )

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=3,
        frameon=True,
        fontsize=9,
    )

    sns.despine()

    save_figure(
        fig,
        output_dir,
        "05_generation_cosine_topk_curve",
    )


# ============================================================
# 14. 保存全局结果
# ============================================================

def save_results(
    trajectory_df,
    method_df,
    step_df,
    metadata_df,
    output_dir,
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    trajectory_path = output_dir / "per_generation_step.csv"
    method_path = output_dir / "aggregate_by_method.csv"
    step_path = output_dir / "aggregate_by_generation_step.csv"
    metadata_path = output_dir / "generation_metadata.csv"

    trajectory_df.to_csv(
        trajectory_path,
        index=False,
        encoding="utf-8",
    )

    method_df.to_csv(
        method_path,
        index=False,
        encoding="utf-8",
    )

    step_df.to_csv(
        step_path,
        index=False,
        encoding="utf-8",
    )

    metadata_df.to_csv(
        metadata_path,
        index=False,
        encoding="utf-8",
    )

    print(f"[Saved] {trajectory_path}")
    print(f"[Saved] {method_path}")
    print(f"[Saved] {step_path}")
    print(f"[Saved] {metadata_path}")


# ============================================================
# 15. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Measure full-generation teacher-forced KL "
            "for CacheClip KV reuse."
        )
    )

    parser.add_argument(
        "--benchmark_jsonl",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--primary_model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
    )

    parser.add_argument(
        "--auxiliary_model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=[
            "bfloat16",
            "float16",
            "float32",
        ],
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--small_attention_last_k",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--repair_ratios",
        type=float,
        nargs="+",
        default=[
            0.10,
            0.20,
            0.30,
            0.40,
        ],
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=12,
    )

    args = parser.parse_args()

    benchmark_path = Path(
        args.benchmark_jsonl
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not benchmark_path.exists():
        raise FileNotFoundError(
            f"Benchmark not found:\n{benchmark_path}"
        )

    if (
        args.device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but CUDA is unavailable."
        )

    dtype = parse_dtype(args.dtype)

    print("=" * 110)
    print(
        "Full-Generation KL Benchmark for CacheClip"
    )
    print("=" * 110)
    print(f"Benchmark: {benchmark_path}")
    print(f"Output: {output_dir}")
    print(f"Primary model: {args.primary_model}")
    print(f"Auxiliary model: {args.auxiliary_model}")
    print(f"Repair ratios: {args.repair_ratios}")
    print(f"Small last K: {args.small_attention_last_k}")
    print("=" * 110)

    records = original.load_jsonl(
        benchmark_path,
        args.max_samples,
    )

    if len(records) == 0:
        raise RuntimeError(
            "No records loaded."
        )

    print(
        f"Loaded samples: {len(records)}"
    )

    print("\nLoading auxiliary model...")

    small_tokenizer, small_model = original.load_model(
        args.auxiliary_model,
        args.device,
        dtype,
    )

    print("Loading primary model...")

    large_tokenizer, large_model = original.load_model(
        args.primary_model,
        args.device,
        dtype,
    )

    small_device = original.get_model_device(
        small_model
    )

    large_device = original.get_model_device(
        large_model
    )

    all_trajectory_rows = []
    all_metadata_rows = []

    success_count = 0

    for sample_index, record in enumerate(
        records,
        start=1,
    ):
        sample_id = record.get(
            "sample_id",
            f"sample_{sample_index:05d}",
        )

        print(
            f"\n[{sample_index}/{len(records)}] "
            f"{sample_id}"
        )

        try:
            (
                trajectory_rows,
                metadata_rows,
            ) = run_one_sample(
                record=record,
                small_tokenizer=small_tokenizer,
                small_model=small_model,
                small_device=small_device,
                large_tokenizer=large_tokenizer,
                large_model=large_model,
                large_device=large_device,
                args=args,
            )

            all_trajectory_rows.extend(
                trajectory_rows
            )

            all_metadata_rows.extend(
                metadata_rows
            )

            success_count += 1

            print(
                f"  trajectory rows: "
                f"{len(trajectory_rows)}"
            )

        except Exception as exc:
            print(
                f"[Warning] Failed sample "
                f"{sample_id}: {exc}"
            )

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(all_trajectory_rows) == 0:
        raise RuntimeError(
            "No trajectory results generated."
        )

    trajectory_df = pd.DataFrame(
        all_trajectory_rows
    )

    metadata_df = pd.DataFrame(
        all_metadata_rows
    )

    step_df = aggregate_by_generation_step(
        trajectory_df
    )

    method_df = aggregate_by_method(
        trajectory_df
    )

    save_results(
        trajectory_df=trajectory_df,
        method_df=method_df,
        step_df=step_df,
        metadata_df=metadata_df,
        output_dir=output_dir,
    )

    # 绘图。
    plot_generation_kl_curve(
        step_df,
        output_dir,
    )

    plot_generation_js_curve(
        step_df,
        output_dir,
    )

    plot_generation_topk_curve(
        step_df,
        output_dir,
    )

    plot_step_heatmap(
        step_df,
        output_dir,
    )

    plot_step_position_summary(
        step_df,
        output_dir,
    )

    config_path = output_dir / "run_config.json"

    config_path.write_text(
        json.dumps(
            vars(args),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[Saved] {config_path}")

    print("\n" + "=" * 110)
    print("Aggregate by Method")
    print("=" * 110)

    print(
        method_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.8f}",
        )
    )

    print("\n" + "=" * 110)
    print("Aggregate by Generation Step")
    print("=" * 110)

    print(
        step_df.head(50).to_string(
            index=False,
            float_format=lambda value: f"{value:.8f}",
        )
    )

    print("\n" + "=" * 110)
    print(
        f"Completed: {success_count}/"
        f"{len(records)} samples"
    )
    print(
        f"Output directory: {output_dir.resolve()}"
    )
    print("=" * 110)


if __name__ == "__main__":
    main()
       
