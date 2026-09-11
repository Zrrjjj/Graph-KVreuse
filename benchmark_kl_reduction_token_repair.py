#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
benchmark_kl_reduction_token_repair.py

KL-Reduction-Guided Token KV Repair Benchmark
==============================================

目的：
    使用 decode-step causal attribution 中的 step_kl_reduction，
    对 Prompt evidence token 排序，选择 Top-ratio token 做 KV patch，
    测试是否比 Full KV Reuse 更有效恢复推理精度。

输入：
    1. Long-path benchmark JSONL
    2. per_prompt_token_decode_attribution.csv
       由 benchmark_decode_step_prompt_token_attribution.py 生成。

比较方法：
    - Full Recompute
    - Full Long-Path KV Reuse
    - KL-Reduction Guided Token Repair 10%
    - KL-Reduction Guided Token Repair 15%
    - KL-Reduction Guided Token Repair 20%
    - KL-Reduction Guided Token Repair 25%
    - KL-Reduction Guided Token Repair 30%

重要：
    step_kl_reduction 来自 Target Full KV Oracle patch attribution。
    因此本实验是 Oracle selector upper-bound / mechanism analysis，
    不能直接视为线上可部署 CacheClip 加速方案。

每个样本中：
    1. 从 attribution CSV 读取每个 evidence token 的 KL reduction；
    2. 按 step_kl_reduction 降序排序；
    3. 选 Top-ratio token；
    4. 将这些 token 的 Source KV 替换为 Target true KV；
    5. 重新 prefill Target path 后 suffix；
    6. 测试生成和 Candidate Ranking 精度。
"""

import re
import gc
import json
import math
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache


# ============================================================
# 1. 基础工具
# ============================================================

def parse_dtype(name):
    name = str(name).lower()

    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16

    if name in {"fp16", "float16"}:
        return torch.float16

    if name in {"fp32", "float32"}:
        return torch.float32

    raise ValueError(f"Unsupported dtype: {name}")


def get_model_device(model):
    return model.get_input_embeddings().weight.device


def sync_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_time(fn, device):
    sync_cuda(device)

    start = time.perf_counter()
    result = fn()

    sync_cuda(device)

    return result, time.perf_counter() - start


def to_legacy_cache(cache):
    if hasattr(cache, "to_legacy_cache"):
        return cache


def normalize_text(text):
    if text is None:
        return ""

    text = str(text).lower()
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_answer_phrase(text):
    if text is None:
        return ""

    text = str(text).strip()

    patterns = [
        r"^\s*answer\s*:\s*",
        r"^\s*the answer is\s+",
        r"^\s*the final entity is\s+",
        r"^\s*the entity is\s+",
        r"^\s*it is\s+",
    ]

    for pattern in patterns:
        text = re.sub(
            pattern,
            "",
            text,
            flags=re.IGNORECASE,
        )

    if "\n" in text:
        text = text.split("\n", 1)[0]

    if "." in text:
        text = text.split(".", 1)[0]

    return text.strip(" \t\r\n\"'`*")


def contains_gold_answer(generated, gold_answers):
    text = normalize_text(generated)

    return int(
        any(
            normalize_text(answer) in text
            for answer in gold_answers
        )
    )


def extracted_entity_em(generated, gold_answers):
    predicted = normalize_text(
        extract_answer_phrase(generated)
    )

    gold_set = {
        normalize_text(answer)
        for answer in gold_answers
    }

    return int(predicted in gold_set)


# ============================================================
# 2. Prompt Span 定位
# ============================================================

def find_question_char_span(prompt):
    question_marker = "Question: "
    answer_marker = "\nAnswer:"

    start = prompt.rfind(question_marker)

    if start < 0:
        raise RuntimeError(
            "Cannot find Question marker."
        )

    start += len(question_marker)

    end = prompt.find(answer_marker, start)

    if end < 0:
        raise RuntimeError(
            "Cannot find Answer marker."
        )

    return start, end


def find_target_path_char_spans(prompt, hop_length):
    path_open_pattern = (
        r"<KG_PATH\s+id=P_TARGET_LONG\s+"
        r"block=TARGET_PATH[^>]*>"
    )

    path_match = re.search(
        path_open_pattern,
        prompt,
    )

    if path_match is None:
        raise RuntimeError(
            "Cannot find target KG_PATH opening tag."
        )

    path_start = path_match.start()

    path_close_tag = "</KG_PATH>"

    path_close_position = prompt.find(
        path_close_tag,
        path_match.end(),
    )

    if path_close_position < 0:
        raise RuntimeError(
            "Cannot find target KG_PATH closing tag."
        )

    path_end = (
        path_close_position
        + len(path_close_tag)
    )

    hop_char_spans = []

    for hop_id in range(1, hop_length + 1):
        triple_pattern = (
            rf"<TRIPLE\s+hop={hop_id}>"
        )

        triple_match = re.search(
            triple_pattern,
            prompt[path_start:path_end],
        )

        if triple_match is None:
            raise RuntimeError(
                f"Cannot find Triple hop={hop_id}."
            )

        triple_start = (
            path_start
            + triple_match.start()
        )

        triple_close_tag = "</TRIPLE>"

        triple_close_position = prompt.find(
            triple_close_tag,
            triple_start,
        )

        if triple_close_position < 0:
            raise RuntimeError(
                f"Cannot find closing tag for Triple hop={hop_id}."
            )

        triple_end = (
            triple_close_position
            + len(triple_close_tag)
        )

        hop_char_spans.append(
            (triple_start, triple_end)
        )

    return {
        "full_path_char_span": (
            path_start,
            path_end,
        ),
        "hop_char_spans": hop_char_spans,
    }


def tokenize_with_offsets(tokenizer, prompt):
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
    )

    return {
        "input_ids": encoded["input_ids"][0].cpu(),
        "offsets": encoded["offset_mapping"][0].tolist(),
    }


def char_span_to_token_positions(
    offsets,
    char_start,
    char_end,
):
    positions = []

    for token_index, (start, end) in enumerate(offsets):
        if start == end:
            continue

        if end > char_start and start < char_end:
            positions.append(token_index)

    if len(positions) == 0:
        raise RuntimeError(
            f"No token overlaps [{char_start}, {char_end})."
        )

    return positions


def extract_prompt_spans(
    tokenizer,
    prompt,
    hop_length,
):
    """
    返回：

    evidence_positions：
        仅包含 <TRIPLE hop=i>...</TRIPLE> 内 token。

    full_path_span：
        包含完整 KG_PATH wrapper 的连续 token span。
    """

    tokenized = tokenize_with_offsets(
        tokenizer,
        prompt,
    )

    input_ids = tokenized["input_ids"]
    offsets = tokenized["offsets"]

    char_info = find_target_path_char_spans(
        prompt,
        hop_length,
    )

    hop_positions = []

    for hop_start, hop_end in char_info[
        "hop_char_spans"
    ]:
        positions = char_span_to_token_positions(
            offsets,
            hop_start,
            hop_end,
        )

        hop_positions.append(positions)

    evidence_positions = sorted(
        {
            token_position
            for positions in hop_positions
            for token_position in positions
        }
    )

    if len(evidence_positions) != sum(
        len(positions)
        for positions in hop_positions
    ):
        raise RuntimeError(
            "Unexpected overlapping Triple spans."
        )

    full_path_positions = char_span_to_token_positions(
        offsets,
        char_info["full_path_char_span"][0],
        char_info["full_path_char_span"][1],
    )

    full_path_span = (
        full_path_positions[0],
        full_path_positions[-1] + 1,
    )

    question_start, question_end = find_question_char_span(
        prompt
    )

    question_positions = char_span_to_token_positions(
        offsets,
        question_start,
        question_end,
    )

    global_to_hop = {}

    for hop_index, positions in enumerate(
        hop_positions,
        start=1,
    ):
        for position in positions:
            global_to_hop[position] = hop_index

    return {
        "input_ids": input_ids,
        "offsets": offsets,
        "evidence_positions": evidence_positions,
        "hop_positions": hop_positions,
        "question_positions": question_positions,
        "full_path_span": full_path_span,
        "global_to_hop": global_to_hop,
    }


# ============================================================
# 3. 模型加载与 Prefill
# ============================================================

def load_model(model_name, device, dtype):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    model.to(device)
    model.eval()

    return tokenizer, model


@torch.no_grad()
def full_prefill(
    model,
    input_ids_cpu,
    device,
):
    input_ids = input_ids_cpu.unsqueeze(0).to(device)

    attention_mask = torch.ones_like(input_ids)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )

    return {
        "logits": outputs.logits[:, -1, :].detach(),
        "cache": to_legacy_cache(
            outputs.past_key_values
        ),
        "seq_len": input_ids.shape[1],
    }


# ============================================================
# 4. Hybrid KV Cache
# ============================================================

def build_hybrid_path_cache(
    source_cache,
    target_cache,
    source_path_span,
    target_path_span,
    target_evidence_positions,
    selected_local_indices,
):
    """
    Hybrid KV:

        Target Prefix Before KG_PATH
        +
        Source Target KG_PATH KV
        +
        selected token Target KV patch
    """

    source_start, source_end = source_path_span
    target_start, target_end = target_path_span

    source_path_length = source_end - source_start
    target_path_length = target_end - target_start

    if source_path_length != target_path_length:
        raise RuntimeError(
            "Source / Target KG_PATH length mismatch."
        )

    selected_set = set(selected_local_indices)

    hybrid_layers = []

    s_k = source_cache.key_cache if hasattr(source_cache, "key_cache") else [l[0] for l in source_cache]
    s_v = source_cache.value_cache if hasattr(source_cache, "value_cache") else [l[1] for l in source_cache]
    t_k = target_cache.key_cache if hasattr(target_cache, "key_cache") else [l[0] for l in target_cache]
    t_v = target_cache.value_cache if hasattr(target_cache, "value_cache") else [l[1] for l in target_cache]

    for source_k, source_v, target_k, target_v in zip(s_k, s_v, t_k, t_v):
        prefix_k = target_k[
            :,
            :,
            :target_start,
            :,
        ].clone()

        prefix_v = target_v[
            :,
            :,
            :target_start,
            :,
        ].clone()

        path_k = source_k[
            :,
            :,
            source_start:source_end,
            :,
        ].clone()

        path_v = source_v[
            :,
            :,
            source_start:source_end,
            :,
        ].clone()

        for local_index in selected_set:
            if not (
                0 <= local_index
                < len(target_evidence_positions)
            ):
                raise RuntimeError(
                    "Patch token index out of range."
                )

            global_position = target_evidence_positions[
                local_index
            ]

            relative_position = (
                global_position - target_start
            )

            if not (
                0 <= relative_position
                < target_path_length
            ):
                raise RuntimeError(
                    "Patch token outside Target KG_PATH."
                )

            path_k[
                :,
                :,
                relative_position,
                :,
            ] = target_k[
                :,
                :,
                global_position,
                :,
            ]

            path_v[
                :,
                :,
                relative_position,
                :,
            ] = target_v[
                :,
                :,
                global_position,
                :,
            ]

        hybrid_layers.append(
            (
                torch.cat(
                    [prefix_k, path_k],
                    dim=2,
                ),
                torch.cat(
                    [prefix_v, path_v],
                    dim=2,
                ),
            )
        )

    new_cache = DynamicCache()
    for k, v in hybrid_layers:
        new_cache.update(k, v, layer_idx=len(new_cache))
    return new_cache


@torch.no_grad()
def recompute_target_suffix(
    model,
    hybrid_cache,
    target_input_ids_cpu,
    target_path_span,
    device,
):
    """
    从 Target KG_PATH 结束位置重新计算：

        Target suffix
        Question
        Answer:
    """

    _, target_path_end = target_path_span

    suffix_ids_cpu = target_input_ids_cpu[
        target_path_end:
    ]

    if len(suffix_ids_cpu) == 0:
        raise RuntimeError(
            "Target suffix is empty."
        )

    prefix_length = target_path_end

    suffix_ids = suffix_ids_cpu.unsqueeze(0).to(device)

    total_length = (
        prefix_length
        + suffix_ids.shape[1]
    )

    attention_mask = torch.ones(
        (1, total_length),
        dtype=torch.long,
        device=device,
    )

    position_ids = torch.arange(
        prefix_length,
        total_length,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    outputs = model(
        input_ids=suffix_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=hybrid_cache,
        use_cache=True,
    )

    return {
        "logits": outputs.logits[:, -1, :].detach(),
        "cache": to_legacy_cache(
            outputs.past_key_values
        ),
        "seq_len": total_length,
    }


# ============================================================
# 5. Generation
# ============================================================

@torch.no_grad()
def greedy_generate(
    model,
    tokenizer,
    initial_logits,
    initial_cache,
    initial_seq_len,
    device,
    max_new_tokens,
):
    logits = initial_logits
    cache = initial_cache

    generated_ids = []

    eos_id = tokenizer.eos_token_id

    for step in range(max_new_tokens):
        next_token = torch.argmax(
            logits,
            dim=-1,
            keepdim=True,
        )

        token_id = int(next_token.item())
        generated_ids.append(token_id)

        if eos_id is not None and token_id == eos_id:
            break

        position = initial_seq_len + step

        attention_mask = torch.ones(
            (1, position + 1),
            dtype=torch.long,
            device=device,
        )

        position_ids = torch.tensor(
            [[position]],
            dtype=torch.long,
            device=device,
        )

        outputs = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )

        cache = to_legacy_cache(
            outputs.past_key_values
        )

        logits = outputs.logits[:, -1, :]

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()


# ============================================================
# 6. Candidate Ranking
# ============================================================

def build_candidates(record, max_candidates):
    candidates = []
    seen = set()

    for answer in record.get("gold_answers", []):
        if answer not in seen:
            candidates.append(answer)
            seen.add(answer)

    for node in record.get(
        "target_path",
        {},
    ).get("nodes", []):
        if node not in seen:
            candidates.append(node)
            seen.add(node)

    for noise_path in record.get("noise_paths", []):
        for node in noise_path.get("nodes", []):
            if node not in seen:
                candidates.append(node)
                seen.add(node)

            if len(candidates) >= max_candidates:
                return candidates

    return candidates[:max_candidates]


@torch.no_grad()
def rank_candidates(
    model,
    tokenizer,
    initial_logits,
    initial_cache,
    initial_seq_len,
    candidates,
    device,
):
    rankings = []

    for candidate in candidates:
        token_ids = tokenizer.encode(
            " " + candidate,
            add_special_tokens=False,
        )

        if len(token_ids) == 0:
            continue

        cache = initial_cache
        logits = initial_logits
        seq_len = initial_seq_len

        total_logprob = 0.0

        for token_index, token_id in enumerate(token_ids):
            log_probs = F.log_softmax(
                logits.float(),
                dim=-1,
            )

            total_logprob += float(
                log_probs[0, token_id].item()
            )

            if token_index == len(token_ids) - 1:
                break

            token = torch.tensor(
                [[token_id]],
                dtype=torch.long,
                device=device,
            )

            attention_mask = torch.ones(
                (1, seq_len + 1),
                dtype=torch.long,
                device=device,
            )

            position_ids = torch.tensor(
                [[seq_len]],
                dtype=torch.long,
                device=device,
            )

            outputs = model(
                input_ids=token,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
            )

            cache = to_legacy_cache(
                outputs.past_key_values
            )

            logits = outputs.logits[:, -1, :]
            seq_len += 1

        rankings.append(
            {
                "candidate": candidate,
                "total_logprob": total_logprob,
                "avg_logprob": (
                    total_logprob
                    / len(token_ids)
                ),
            }
        )

    return sorted(
        rankings,
        key=lambda item: item["avg_logprob"],
        reverse=True,
    )


# ============================================================
# 7. Metrics
# ============================================================

def calculate_logits_metrics(
    full_logits,
    method_logits,
    top_k,
):
    full_logits = full_logits.float()
    method_logits = method_logits.float()

    full_probs = F.softmax(
        full_logits,
        dim=-1,
    )

    method_log_probs = F.log_softmax(
        method_logits,
        dim=-1,
    )

    kl = F.kl_div(
        method_log_probs,
        full_probs,
        reduction="batchmean",
    ).item()

    if abs(kl) < 1e-8:
        kl = 0.0

    cosine = F.cosine_similarity(
        full_logits,
        method_logits,
        dim=-1,
    ).mean().item()

    full_top1 = int(
        torch.argmax(full_logits, dim=-1).item()
    )

    method_top1 = int(
        torch.argmax(method_logits, dim=-1).item()
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

    return {
        "next_token_kl": kl,
        "next_token_logits_cosine": cosine,
        "next_token_top1_match": int(
            full_top1 == method_top1
        ),
        "next_token_topk_overlap": (
            len(full_topk & method_topk)
            / float(top_k)
        ),
    }


def gold_margin(rankings, gold_answers):
    gold_set = {
        normalize_text(answer)
        for answer in gold_answers
    }

    gold_scores = []
    wrong_scores = []

    gold_rank = None

    for rank, item in enumerate(rankings, start=1):
        score = float(item["avg_logprob"])
        if normalize_text(item["candidate"]) in gold_set:
            gold_scores.append(score)

            if gold_rank is None:
                gold_rank = rank
        else:
            wrong_scores.append(score)

    if len(gold_scores) == 0:
        return {
            "gold_score": float("-inf"),
            "best_wrong_score": float("-inf"),
            "gold_margin": float("-inf"),
            "gold_rank": None,
        }

    gold_score = max(gold_scores)

    best_wrong_score = (
        max(wrong_scores)
        if len(wrong_scores) > 0
        else float("-inf")
    )

    if np.isfinite(best_wrong_score):
        margin = gold_score - best_wrong_score
    else:
        margin = gold_score

    return {
        "gold_score": gold_score,
        "best_wrong_score": best_wrong_score,
        "gold_margin": margin,
        "gold_rank": gold_rank,
    }


def evaluate_state(
    record,
    state,
    full_state,
    model,
    tokenizer,
    candidates,
    device,
    top_k,
    max_new_tokens,
):
    """
    对一个 Prompt KV state 评估：
        - Candidate Accuracy
        - Gold Margin
        - Contains Gold Answer
        - Extracted Entity EM
        - Next-token distribution fidelity
    """

    gold_answers = record.get(
        "gold_answers",
        [],
    )

    rankings = rank_candidates(
        model=model,
        tokenizer=tokenizer,
        initial_logits=state["logits"],
        initial_cache=state["cache"],
        initial_seq_len=state["seq_len"],
        candidates=candidates,
        device=device,
    )

    margin_result = gold_margin(
        rankings,
        gold_answers,
    )

    prediction = (
        rankings[0]["candidate"]
        if len(rankings) > 0
        else ""
    )

    gold_set = {
        normalize_text(answer)
        for answer in gold_answers
    }

    candidate_accuracy = int(
        normalize_text(prediction) in gold_set
    )

    generated_answer = greedy_generate(
        model=model,
        tokenizer=tokenizer,
        initial_logits=state["logits"],
        initial_cache=state["cache"],
        initial_seq_len=state["seq_len"],
        device=device,
        max_new_tokens=max_new_tokens,
    )

    logits_metrics = calculate_logits_metrics(
        full_logits=full_state["logits"],
        method_logits=state["logits"],
        top_k=top_k,
    )

    return {
        "candidate_prediction": prediction,
        "candidate_accuracy": candidate_accuracy,
        "candidate_rankings": json.dumps(
            rankings,
            ensure_ascii=False,
        ),
        "gold_score": margin_result["gold_score"],
        "best_wrong_score": margin_result[
            "best_wrong_score"
        ],
        "gold_margin": margin_result["gold_margin"],
        "gold_rank": margin_result["gold_rank"],
        "generated_answer": generated_answer,
        "contains_gold_answer": contains_gold_answer(
            generated_answer,
            gold_answers,
        ),
        "extracted_entity_em": extracted_entity_em(
            generated_answer,
            gold_answers,
        ),
        "generation_match_full": None,
        **logits_metrics,
    }


# ============================================================
# 8. Attribution CSV 读取与对齐
# ============================================================

def load_attribution_data(
    attribution_csv,
    target_generation_step=None,
):
    """
    读取前一个 decode-step attribution 实验得到的：

        per_prompt_token_decode_attribution.csv

    必需列：
        sample_id
        path_local_index
        step_kl_reduction
        was_attributed

    如果 attribution 使用的是 Step 8，
    本实验也建议仅使用对应 Step 8 的结果。
    """

    attribution_path = Path(attribution_csv)

    if not attribution_path.exists():
        raise FileNotFoundError(
            f"Attribution CSV not found:\n{attribution_path}"
        )

    df = pd.read_csv(attribution_path)

    required_columns = {
        "sample_id",
        "path_local_index",
        "step_kl_reduction",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise RuntimeError(
            "Attribution CSV missing columns:\n"
            f"{sorted(missing)}"
        )

    if "was_attributed" in df.columns:
        df = df[
            pd.to_numeric(
                df["was_attributed"],
                errors="coerce",
            ).fillna(0).astype(int)
            == 1
        ].copy()

    if (
        target_generation_step is not None
        and "target_generation_step" in df.columns
    ):
        df = df[
            pd.to_numeric(
                df["target_generation_step"],
                errors="coerce",
            )
            == int(target_generation_step)
        ].copy()

    df["path_local_index"] = pd.to_numeric(
        df["path_local_index"],
        errors="coerce",
    )

    df["step_kl_reduction"] = pd.to_numeric(
        df["step_kl_reduction"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "sample_id",
            "path_local_index",
            "step_kl_reduction",
        ]
    ).copy()

    df["path_local_index"] = df[
        "path_local_index"
    ].astype(int)

    # 如果同一个 sample/local index 有重复记录，保留最后或均值。
    # 实际上应当每个 token 一行。
    df = (
        df.groupby(
            [
                "sample_id",
                "path_local_index",
            ],
            as_index=False,
        )
        .agg(
            step_kl_reduction=(
                "step_kl_reduction",
                "mean",
            ),
            **{
                column: (column, "first")
                for column in [
                    "hop_index",
                    "token_type",
                    "token_label",
                    "key_relative_deviation_mean",
                    "value_relative_deviation_mean",
                    "raw_kv_deviation_score",
                ]
                if column in df.columns
            },
        )
    )

    return df


def select_kl_reduction_tokens(
    attribution_df,
    sample_id,
    total_evidence_tokens,
    repair_ratio,
    negative_policy="allow",
):
    """
    对当前 sample 的 token 按 step_kl_reduction 降序选择。

    参数：
        negative_policy:
            allow:
                即使后面 token 的 KL reduction 为负，
                仍按固定预算选择 top-ratio。

            positive_only:
                仅选择 step_kl_reduction > 0 的 token。
                这时实际 selected ratio 可能低于 requested ratio。

    返回：
        selected_indices
        requested_budget
        available_attribution_count
        selected_score_stats
    """

    requested_budget = max(
        1,
        int(
            math.ceil(
                total_evidence_tokens
                * repair_ratio
            )
        ),
    )

    sample_attr = attribution_df[
        attribution_df["sample_id"] == sample_id
    ].copy()

    if len(sample_attr) == 0:
        return [], requested_budget, 0, {
            "selected_mean_kl_reduction": np.nan,
            "selected_min_kl_reduction": np.nan,
            "selected_max_kl_reduction": np.nan,
        }

    # 必须限制在当前 path 的合法 token index 内。
    sample_attr = sample_attr[
        (
            sample_attr["path_local_index"] >= 0
        )
        & (
            sample_attr["path_local_index"]
            < total_evidence_tokens
        )
    ].copy()

    if negative_policy == "positive_only":
        sample_attr = sample_attr[
            sample_attr["step_kl_reduction"] > 0
        ].copy()

    elif negative_policy != "allow":
        raise ValueError(
            "negative_policy must be allow or positive_only."
        )

    sample_attr = sample_attr.sort_values(
        "step_kl_reduction",
        ascending=False,
    )

    selected = sample_attr.head(
        min(
            requested_budget,
            len(sample_attr),
        )
    ).copy()

    selected_indices = sorted(
        selected["path_local_index"]
        .astype(int)
        .tolist()
    )

    if len(selected) == 0:
        score_stats = {
            "selected_mean_kl_reduction": np.nan,
            "selected_min_kl_reduction": np.nan,
            "selected_max_kl_reduction": np.nan,
        }
    else:
        score_stats = {
            "selected_mean_kl_reduction": float(
                selected[
                    "step_kl_reduction"
                ].mean()
            ),
            "selected_min_kl_reduction": float(
                selected[
                    "step_kl_reduction"
                ].min()
            ),
            "selected_max_kl_reduction": float(
                selected[
                    "step_kl_reduction"
                ].max()
            ),
        }

    return (
        selected_indices,
        requested_budget,
        len(sample_attr),
        score_stats,
    )


# ============================================================
# 9. 单样本 KL-Reduction-Guided Repair
# ============================================================

def run_one_sample(
    record,
    tokenizer,
    model,
    device,
    attribution_df,
    args,
):
    sample_id = record["sample_id"]
    hop_length = int(record["hop_length"])

    source_prompt = record["source_prompt"]
    target_prompt = record["target_prompt"]

    # --------------------------------------------------------
    # A. Prompt alignment
    # --------------------------------------------------------
    source_spans = extract_prompt_spans(
        tokenizer,
        source_prompt,
        hop_length,
    )

    target_spans = extract_prompt_spans(
        tokenizer,
        target_prompt,
        hop_length,
    )

    source_evidence_positions = source_spans[
        "evidence_positions"
    ]

    target_evidence_positions = target_spans[
        "evidence_positions"
    ]

    if len(source_evidence_positions) != len(
        target_evidence_positions
    ):
        raise RuntimeError(
            "Source/Target evidence token count mismatch."
        )

    source_evidence_ids = source_spans[
        "input_ids"
    ][source_evidence_positions]

    target_evidence_ids = target_spans[
        "input_ids"
    ][target_evidence_positions]

    if not torch.equal(
        source_evidence_ids,
        target_evidence_ids,
    ):
        raise RuntimeError(
            "Source/Target evidence token IDs differ."
        )

    source_path_start, source_path_end = source_spans[
        "full_path_span"
    ]

    target_path_start, target_path_end = target_spans[
        "full_path_span"
    ]

    source_path_ids = source_spans[
        "input_ids"
    ][source_path_start:source_path_end]

    target_path_ids = target_spans[
        "input_ids"
    ][target_path_start:target_path_end]

    if not torch.equal(
        source_path_ids,
        target_path_ids,
    ):
        raise RuntimeError(
            "Source/Target full KG_PATH token IDs differ."
        )

    evidence_token_count = len(
        target_evidence_positions
    )

    # --------------------------------------------------------
    # B. Full Source / Target Prefill
    # --------------------------------------------------------
    source_full, source_prefill_latency = measure_time(
        lambda: full_prefill(
            model=model,
            input_ids_cpu=source_spans["input_ids"],
            device=device,
        ),
        device,
    )

    target_full, target_prefill_latency = measure_time(
        lambda: full_prefill(
            model=model,
            input_ids_cpu=target_spans["input_ids"],
            device=device,
        ),
        device,
    )

    candidates = build_candidates(
        record,
        args.max_candidates,
    )

    # --------------------------------------------------------
    # C. Full Recompute baseline
    # --------------------------------------------------------
    full_eval, full_eval_latency = measure_time(
        lambda: evaluate_state(
            record=record,
            state=target_full,
            full_state=target_full,
            model=model,
            tokenizer=tokenizer,
            candidates=candidates,
            device=device,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
        ),
        device,
    )

    full_eval["generation_match_full"] = 1

    # --------------------------------------------------------
    # D. Full Long-Path KV Reuse baseline
    # --------------------------------------------------------
    reuse_cache, reuse_patch_latency = measure_time(
        lambda: build_hybrid_path_cache(
            source_cache=source_full["cache"],
            target_cache=target_full["cache"],
            source_path_span=source_spans[
                "full_path_span"
            ],
            target_path_span=target_spans[
                "full_path_span"
            ],
            target_evidence_positions=(
                target_evidence_positions
            ),
            selected_local_indices=[],
        ),
        device,
    )

    reuse_state, reuse_suffix_latency = measure_time(
        lambda: recompute_target_suffix(
            model=model,
            hybrid_cache=reuse_cache,
            target_input_ids_cpu=target_spans[
                "input_ids"
            ],
            target_path_span=target_spans[
                "full_path_span"
            ],
            device=device,
        ),
        device,
    )

    reuse_eval, reuse_eval_latency = measure_time(
        lambda: evaluate_state(
            record=record,
            state=reuse_state,
            full_state=target_full,
            model=model,
            tokenizer=tokenizer,
            candidates=candidates,
            device=device,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
        ),
        device,
    )

    reuse_eval["generation_match_full"] = int(
        normalize_text(
            reuse_eval["generated_answer"]
        )
        == normalize_text(
            full_eval["generated_answer"]
        )
    )

    result_rows = []

    # Full Recompute result
    result_rows.append(
        {
            "sample_id": sample_id,
            "scenario": record.get(
                "scenario",
                "",
            ),
            "hop_length": hop_length,
            "method": "Full Recompute",
            "method_type": "full_recompute",
            "selector_type": "none",
            "repair_ratio": 1.0,
            "requested_token_budget": evidence_token_count,
            "selected_token_count": evidence_token_count,
            "actual_repair_ratio": 1.0,
            "available_attribution_token_count": evidence_token_count,
            "selected_local_indices": json.dumps(
                list(range(evidence_token_count))
            ),
            "selected_mean_kl_reduction": np.nan,
            "selected_min_kl_reduction": np.nan,
            "selected_max_kl_reduction": np.nan,
            **full_eval,
            "source_prefill_latency_sec": 0.0,
            "target_prefill_latency_sec": (
                target_prefill_latency
            ),
            "patch_latency_sec": 0.0,
            "suffix_latency_sec": 0.0,
            "evaluation_latency_sec": full_eval_latency,
            "total_latency_sec": (
                target_prefill_latency
                + full_eval_latency
            ),
        }
    )

    # Full Reuse result
    result_rows.append(
        {
            "sample_id": sample_id,
            "scenario": record.get(
                "scenario",
                "",
            ),
            "hop_length": hop_length,
            "method": "Full Long-Path KV Reuse",
            "method_type": "full_reuse",
            "selector_type": "none",
            "repair_ratio": 0.0,
            "requested_token_budget": 0,
            "selected_token_count": 0,
            "actual_repair_ratio": 0.0,
            "available_attribution_token_count": 0,
            "selected_local_indices": json.dumps([]),
            "selected_mean_kl_reduction": np.nan,
            "selected_min_kl_reduction": np.nan,
            "selected_max_kl_reduction": np.nan,
            **reuse_eval,
            "source_prefill_latency_sec": (
                source_prefill_latency
            ),
            "target_prefill_latency_sec": (
                target_prefill_latency
            ),
            "patch_latency_sec": (
                reuse_patch_latency
            ),
            "suffix_latency_sec": (
                reuse_suffix_latency
            ),
            "evaluation_latency_sec": (
                reuse_eval_latency
            ),
            "total_latency_sec": (
                source_prefill_latency
                + target_prefill_latency
                + reuse_patch_latency
                + reuse_suffix_latency
                + reuse_eval_latency
            ),
        }
    )

    # --------------------------------------------------------
    # E. KL-reduction-guided repair sweep
    # --------------------------------------------------------
    selected_token_rows = []

    for repair_ratio in args.repair_ratios:
        (
            selected_indices,
            requested_budget,
            available_attribution_count,
            selected_score_stats,
        ) = select_kl_reduction_tokens(
            attribution_df=attribution_df,
            sample_id=sample_id,
            total_evidence_tokens=evidence_token_count,
            repair_ratio=repair_ratio,
            negative_policy=args.negative_policy,
        )

        if len(selected_indices) == 0:
            print(
                f"[Warning] Sample {sample_id}: no attributed token "
                f"available for repair ratio {repair_ratio:.0%}."
            )

            # 仍记录一个空 selection；结果等价于 reuse。
            selected_state = reuse_state
            selected_patch_latency = 0.0
            selected_suffix_latency = 0.0

        else:
            selected_cache, selected_patch_latency = measure_time(
                lambda: build_hybrid_path_cache(
                    source_cache=source_full["cache"],
                    target_cache=target_full["cache"],
                    source_path_span=source_spans[
                        "full_path_span"
                    ],
                    target_path_span=target_spans[
                        "full_path_span"
                    ],
                    target_evidence_positions=(
                        target_evidence_positions
                    ),
                    selected_local_indices=selected_indices,
                ),
                device,
            )

            selected_state, selected_suffix_latency = measure_time(
                lambda: recompute_target_suffix(
                    model=model,
                    hybrid_cache=selected_cache,
                    target_input_ids_cpu=target_spans[
                        "input_ids"
                    ],
                    target_path_span=target_spans[
                        "full_path_span"
                    ],
                    device=device,
                ),
                device,
            )

        selected_eval, selected_eval_latency = measure_time(
            lambda: evaluate_state(
                record=record,
                state=selected_state,
                full_state=target_full,
                model=model,
                tokenizer=tokenizer,
                candidates=candidates,
                device=device,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
            ),
            device,
        )

        selected_eval["generation_match_full"] = int(
            normalize_text(
                selected_eval["generated_answer"]
            )
            == normalize_text(
                full_eval["generated_answer"]
            )
        )

        actual_repair_ratio = (
            len(selected_indices)
            / float(max(evidence_token_count, 1))
        )

        result_rows.append(
            {
                "sample_id": sample_id,
                "scenario": record.get(
                    "scenario",
                    "",
                ),
                "hop_length": hop_length,
                "method": (
                    "KL-Reduction Guided Token Repair "
                    f"{repair_ratio:.0%}"
                ),
                "method_type": "kl_reduction_guided_patch",
                "selector_type": "decode_step_kl_reduction",
                "repair_ratio": repair_ratio,
                "requested_token_budget": requested_budget,
                "selected_token_count": len(
                    selected_indices
                ),
                "actual_repair_ratio": actual_repair_ratio,
                "available_attribution_token_count": (
                    available_attribution_count
                ),
                "selected_local_indices": json.dumps(
                    selected_indices
                ),
                **selected_score_stats,
                **selected_eval,
                "source_prefill_latency_sec": (
                    source_prefill_latency
                ),
                "target_prefill_latency_sec": (
                    target_prefill_latency
                ),
                "patch_latency_sec": (
                    selected_patch_latency
                ),
                "suffix_latency_sec": (
                    selected_suffix_latency
                ),
                "evaluation_latency_sec": (
                    selected_eval_latency
                ),
                "total_latency_sec": (
                    source_prefill_latency
                    + target_prefill_latency
                    + selected_patch_latency
                    + selected_suffix_latency
                    + selected_eval_latency
                ),
            }
        )

        # 记录本 method 每个 token 是否被选中，便于后续分析。
        selected_set = set(selected_indices)

        sample_attr = attribution_df[
            attribution_df["sample_id"] == sample_id
        ].copy()

        score_map = {
            int(row["path_local_index"]): float(
                row["step_kl_reduction"]
            )
            for _, row in sample_attr.iterrows()
            if 0 <= int(row["path_local_index"]) < evidence_token_count
        }

        target_ids_list = target_spans[
            "input_ids"
        ].tolist()

        for local_index, global_position in enumerate(
            target_evidence_positions
        ):
            token_id = int(
                target_ids_list[global_position]
            )

            token_label = tokenizer.convert_ids_to_tokens(
                token_id
            )

            if token_label is None:
                token_label = str(token_id)

            selected_token_rows.append(
                {
                    "sample_id": sample_id,
                    "repair_ratio": repair_ratio,
                    "path_local_index": local_index,
                    "global_prompt_position": global_position,
                    "token_id": token_id,
                    "token_label": token_label.replace(
                        "\n",
                        "\\n",
                    ),
                    "hop_index": target_spans[
                        "global_to_hop"
                    ][global_position],
                    "step_kl_reduction": score_map.get(
                        local_index,
                        np.nan,
                    ),
                    "was_attributed": int(
                        local_index in score_map
                    ),
                    "selected": int(
                        local_index in selected_set
                    ),
                }
            )

        if len(selected_indices) > 0:
            del selected_state

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # F. Metadata
    # --------------------------------------------------------
    metadata = {
        "sample_id": sample_id,
        "hop_length": hop_length,
        "evidence_token_count": evidence_token_count,
        "source_path_start": source_path_start,
        "target_path_start": target_path_start,
        "path_position_shift": (
            target_path_start
            - source_path_start
        ),
        "attribution_step": args.target_generation_step,
        "available_attribution_token_count": int(
            len(
                attribution_df[
                    attribution_df["sample_id"]
                    == sample_id
                ]
            )
        ),
        "full_generated_answer": full_eval[
            "generated_answer"
        ],
        "reuse_generated_answer": reuse_eval[
            "generated_answer"
        ],
        "full_candidate_prediction": full_eval[
            "candidate_prediction"
        ],
        "reuse_candidate_prediction": reuse_eval[
            "candidate_prediction"
        ],
    }

    del source_full
    del target_full
    del reuse_state

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        pd.DataFrame(result_rows),
        pd.DataFrame(selected_token_rows),
        metadata,
    )


# ============================================================
# 10. 聚合
# ============================================================

def aggregate_results(result_df):
    grouped = (
        result_df.groupby(
            [
                "method",
                "method_type",
                "selector_type",
                "repair_ratio",
            ],
            as_index=False,
        )
        .agg(
            n_samples=(
                "sample_id",
                "nunique",
            ),

            candidate_accuracy=(
                "candidate_accuracy",
                "mean",
            ),

            contains_gold_answer=(
                "contains_gold_answer",
                "mean",
            ),

            extracted_entity_em=(
                "extracted_entity_em",
                "mean",
            ),

            generation_match_full=(
                "generation_match_full",
                "mean",
            ),

            mean_gold_score=(
                "gold_score",
                "mean",
            ),

            mean_best_wrong_score=(
                "best_wrong_score",
                "mean",
            ),

            mean_gold_margin=(
                "gold_margin",
                "mean",
            ),

            mean_gold_rank=(
                "gold_rank",
                "mean",
            ),

            mean_next_token_kl=(
                "next_token_kl",
                "mean",
            ),

            mean_logits_cosine=(
                "next_token_logits_cosine",
                "mean",
            ),

            mean_top1_match=(
                "next_token_top1_match",
                "mean",
            ),

            mean_topk_overlap=(
                "next_token_topk_overlap",
                "mean",
            ),

            mean_requested_token_budget=(
                "requested_token_budget",
                "mean",
            ),

            mean_selected_token_count=(
                "selected_token_count",
                "mean",
            ),

            mean_actual_repair_ratio=(
                "actual_repair_ratio",
                "mean",
            ),

            mean_available_attribution_token_count=(
                "available_attribution_token_count",
                "mean",
            ),

            mean_selected_kl_reduction=(
                "selected_mean_kl_reduction",
                "mean",
            ),

            mean_total_latency_sec=(
                "total_latency_sec",
                "mean",
            ),
        )
    )

    def order(row):
        method_type = row["method_type"]

        if method_type == "full_reuse":
            return 0.0

        if method_type == "kl_reduction_guided_patch":
            return 1.0 + float(
                row["repair_ratio"]
            )

        if method_type == "full_recompute":
            return 3.0

        return 99.0

    grouped["method_order"] = grouped.apply(
        order,
        axis=1,
    )

    grouped = grouped.sort_values(
        [
            "method_order",
            "repair_ratio",
        ]
    ).reset_index(drop=True)

    # Full Reuse -> Full Recompute recovery
    reuse_rows = grouped[
        grouped["method_type"] == "full_reuse"
    ]

    full_rows = grouped[
        grouped["method_type"] == "full_recompute"
    ]

    metrics = [
        "candidate_accuracy",
        "contains_gold_answer",
        "extracted_entity_em",
        "generation_match_full",
        "mean_gold_margin",
        "mean_topk_overlap",
        "mean_logits_cosine",
    ]

    if len(reuse_rows) > 0 and len(full_rows) > 0:
        reuse = reuse_rows.iloc[0]
        full = full_rows.iloc[0]

        for metric in metrics:
            reuse_value = float(reuse[metric])
            full_value = float(full[metric])

            denominator = full_value - reuse_value

            col = f"{metric}_recovery"

            if abs(denominator) < 1e-12:
                grouped[col] = np.nan
            else:
                grouped[col] = (
                    grouped[metric] - reuse_value
                ) / denominator

    return grouped


def aggregate_selection(selected_token_df):
    if len(selected_token_df) == 0:
        return pd.DataFrame()

    return (
        selected_token_df.groupby(
            [
                "repair_ratio",
                "hop_index",
            ],
            as_index=False,
        )
        .agg(
            n_tokens=(
                "path_local_index",
                "count",
            ),

            selected_token_rate=(
                "selected",
                "mean",
            ),

            mean_kl_reduction_score=(
                "step_kl_reduction",
                "mean",
            ),

            selected_attributed_token_rate=(
                "was_attributed",
                "mean",
            ),
        )
    )


# ============================================================
# 11. 可视化
# ============================================================

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


def plot_accuracy_recovery(
    summary_df,
    output_dir,
):
    """
    画 Candidate Accuracy / Contains Gold / Gold Margin。
    """

    reuse = summary_df[
        summary_df["method_type"] == "full_reuse"
    ]

    full = summary_df[
        summary_df["method_type"] == "full_recompute"
    ]

    guided = summary_df[
        summary_df["method_type"]
        == "kl_reduction_guided_patch"
    ].sort_values("repair_ratio")

    if len(guided) == 0:
        return

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17.0, 5.5),
        constrained_layout=True,
    )

    specs = [
        (
            "candidate_accuracy",
            "Candidate Accuracy",
            "Accuracy (%)",
            True,
            "#1B9E77",
        ),
        (
            "contains_gold_answer",
            "Contains Gold Answer",
            "Contains Gold (%)",
            True,
            "#377EB8",
        ),
        (
            "mean_gold_margin",
            "Gold Candidate Margin",
            "Gold Margin",
            False,
            "#7570B3",
        ),
    ]

    x = guided["repair_ratio"].to_numpy() * 100.0

    for ax, (
        metric,
        title,
        ylabel,
        is_percent,
        color,
    ) in zip(axes, specs):
        y = guided[metric].to_numpy()

        if is_percent:
            y = y * 100.0

        ax.plot(
            x,
            y,
            color=color,
            marker="o",
            label="KL-Reduction Guided Repair",
        )

        if len(reuse) > 0:
            reuse_value = float(
                reuse.iloc[0][metric]
            )

            if is_percent:
                reuse_value *= 100.0

            ax.scatter(
                [0],
                [reuse_value],
                marker="D",
                s=75,
                color="#D95F02",
                edgecolor="black",
                linewidth=0.7,
                label="Full KV Reuse",
                zorder=5,
            )

            ax.axhline(
                reuse_value,
                color="#D95F02",
                linestyle=":",
                linewidth=1.0,
                alpha=0.7,
            )

        if len(full) > 0:
            full_value = float(
                full.iloc[0][metric]
            )

            if is_percent:
                full_value *= 100.0

            ax.scatter(
                [100],
                [full_value],
                marker="*",
                s=160,
                color="#1B9E77",
                edgecolor="black",
                linewidth=0.7,
                label="Full Recompute",
                zorder=5,
            )

            ax.axhline(
                full_value,
                color="#1B9E77",
                linestyle=":",
                linewidth=1.0,
                alpha=0.7,
            )

        ax.set_title(
            title,
            fontweight="bold",
        )

        ax.set_xlabel(
            "KL-Reduction-Guided Token Repair Ratio (%)"
        )

        ax.set_ylabel(ylabel)

        ax.set_xlim(-4, 104)

        if is_percent:
            ax.set_ylim(0, 105)
            ax.yaxis.set_major_formatter(
                mtick.PercentFormatter()
            )

        ax.grid(
            axis="y",
            alpha=0.25,
        )

        sns.despine(ax=ax)

    handles, labels = axes[0].get_legend_handles_labels()

    unique = {}

    for handle, label in zip(handles, labels):
        if label not in unique:
            unique[label] = handle

    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=3,
        frameon=True,
    )

    save_figure(
        fig,
        output_dir,
        "01_kl_reduction_guided_accuracy_recovery",
    )


def plot_fidelity_recovery(
    summary_df,
    output_dir,
):
    """
    画 next-token KL、Cosine、Top-k overlap。
    """

    reuse = summary_df[
        summary_df["method_type"] == "full_reuse"
    ]

    full = summary_df[
        summary_df["method_type"] == "full_recompute"
    ]

    guided = summary_df[
        summary_df["method_type"]
        == "kl_reduction_guided_patch"
    ].sort_values("repair_ratio")

    if len(guided) == 0:
        return

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17.0, 5.5),
        constrained_layout=True,
    )

    specs = [
        (
            "mean_next_token_kl",
            "Next-token KL Divergence",
            "KL (lower is better)",
            False,
            "#D95F02",
        ),
        (
            "mean_logits_cosine",
            "Logits Cosine Similarity",
            "Cosine Similarity",
            False,
            "#4C78A8",
        ),
        (
            "mean_topk_overlap",
            "Next-token Top-k Overlap",
            "Top-k Overlap (%)",
            True,
            "#7570B3",
        ),
    ]

    x = guided["repair_ratio"].to_numpy() * 100.0

    for ax, (
        metric,
        title,
        ylabel,
        is_percent,
        color,
    ) in zip(axes, specs):
        y = guided[metric].to_numpy()

        if is_percent:
            y = y * 100.0

        ax.plot(
            x,
            y,
            color=color,
            marker="o",
            label="KL-Reduction Guided Repair",
        )

        if len(reuse) > 0:
            reuse_value = float(
                reuse.iloc[0][metric]
            )

            if is_percent:
                reuse_value *= 100.0

            ax.scatter(
                [0],
                [reuse_value],
                marker="D",
                s=75,
                color="#D95F02",
                edgecolor="black",
                linewidth=0.7,
                label="Full KV Reuse",
                zorder=5,
            )

        if len(full) > 0:
            full_value = float(
                full.iloc[0][metric]
            )

            if is_percent:
                full_value *= 100.0

            ax.scatter(
                [100],
                [full_value],
                marker="*",
                s=160,
                color="#1B9E77",
                edgecolor="black",
                linewidth=0.7,
                label="Full Recompute",
                zorder=5,
            )

        ax.set_title(
            title,
            fontweight="bold",
        )

        ax.set_xlabel("Repair Ratio (%)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-4, 104)

        if is_percent:
            ax.set_ylim(0, 105)
            ax.yaxis.set_major_formatter(
                mtick.PercentFormatter()
            )

        sns.despine(ax=ax)

    handles, labels = axes[0].get_legend_handles_labels()

    unique = {}

    for handle, label in zip(handles, labels):
        if label not in unique:
            unique[label] = handle

    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=3,
        frameon=True,
    )

    save_figure(
        fig,
        output_dir,
        "02_kl_reduction_guided_fidelity_recovery",
    )


def plot_hop_selection(
    selection_df,
    output_dir,
):
    """
    展示 KL-reduction selector 在不同预算下，
    哪些 Hop 的 token 被更多选中。
    """

    if len(selection_df) == 0:
        return

    repair_ratios = sorted(
        selection_df["repair_ratio"].unique()
    )

    hops = sorted(
        selection_df["hop_index"].unique()
    )

    fig, ax = plt.subplots(
        figsize=(11.5, 6.0),
    )

    x = np.arange(len(hops))

    width = 0.8 / max(
        len(repair_ratios),
        1,
    )

    colors = sns.color_palette(
        "Purples",
        n_colors=len(repair_ratios) + 2,
    )[2:]

    for index, ratio in enumerate(repair_ratios):
        subset = selection_df[
            selection_df["repair_ratio"] == ratio
        ].set_index("hop_index")

        values = []

        for hop in hops:
            if hop in subset.index:
                values.append(
                    subset.loc[
                        hop,
                        "selected_token_rate",
                    ]
                    * 100.0
                )
            else:
                values.append(0.0)

        offset = (
            index - (len(repair_ratios) - 1) / 2.0
        ) * width

        ax.bar(
            x + offset,
            values,
            width=width,
            color=colors[index],
            edgecolor="white",
            linewidth=0.55,
            label=f"Repair {ratio:.0%}",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"Hop {int(hop)}"
            for hop in hops
        ]
    )

    ax.set_xlabel("KG Path Hop")
    ax.set_ylabel("Selected Token Rate within Hop (%)")

    ax.set_ylim(0, 105)

    ax.yaxis.set_major_formatter(
        mtick.PercentFormatter()
    )

    ax.set_title(
        "Where Does KL-Reduction-Guided Repair Allocate Its Budget?",
        pad=12,
        fontweight="bold",
    )

    ax.legend(
        frameon=True,
        ncol=min(5, len(repair_ratios)),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
    )

    sns.despine(ax=ax)

    save_figure(
        fig,
        output_dir,
        "03_kl_reduction_guided_hop_selection",
    )


# ============================================================
# 12. JSONL Loader
# ============================================================

def load_jsonl(path, max_samples):
    records = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(line)
            )

            if len(records) >= max_samples:
                break

    return records


# ============================================================
# 13. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate KL-reduction-guided token KV repair."
        )
    )

    parser.add_argument(
        "--benchmark_jsonl",
        type=str,
        required=True,
        help="Long-path benchmark JSONL.",
    )

    parser.add_argument(
        "--attribution_csv",
        type=str,
        required=True,
        help=(
            "per_prompt_token_decode_attribution.csv "
            "from the previous decode-step attribution experiment."
        ),
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
        "--target_generation_step",
        type=int,
        default=None,
        help=(
            "Optional filter. Must equal the step used to build "
            "the attribution CSV, e.g. 8."
        ),
    )

    parser.add_argument(
        "--repair_ratios",
        type=float,
        nargs="+",
        default=[
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
        ],
    )

    parser.add_argument(
        "--negative_policy",
        type=str,
        default="allow",
        choices=[
            "allow",
            "positive_only",
        ],
        help=(
            "allow: always select exactly Top-ratio tokens. "
            "positive_only: select only tokens whose single-token "
            "step_kl_reduction is positive."
        ),
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

    parser.add_argument(
        "--max_candidates",
        type=int,
        default=12,
    )

    args = parser.parse_args()

    for ratio in args.repair_ratios:
        if not (0 < ratio < 1):
            raise ValueError(
                "repair_ratios must be strictly between 0 and 1."
            )

    benchmark_path = Path(
        args.benchmark_jsonl
    )

    output_dir = Path(
        args.output_dir
    )

    figure_dir = output_dir / "figures"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_dir.mkdir(
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
            "CUDA requested but unavailable."
        )

    dtype = parse_dtype(args.dtype)

    setup_paper_style()

    print("=" * 120)
    print(
        "KL-Reduction-Guided Token KV Repair Benchmark"
    )
    print("=" * 120)
    print(f"Benchmark: {benchmark_path}")
    print(f"Attribution CSV: {args.attribution_csv}")
    print(f"Output: {output_dir}")
    print(f"Primary model: {args.primary_model}")
    print(f"Repair ratios: {args.repair_ratios}")
    print(
        f"Negative selection policy: "
        f"{args.negative_policy}"
    )
    print("=" * 120)

    attribution_df = load_attribution_data(
        attribution_csv=args.attribution_csv,
        target_generation_step=args.target_generation_step,
    )

    print(
        f"Loaded attribution rows: {len(attribution_df)}"
    )

    print(
        f"Attribution samples: "
        f"{attribution_df['sample_id'].nunique()}"
    )

    records = load_jsonl(
        benchmark_path,
        args.max_samples,
    )

    if len(records) == 0:
        raise RuntimeError(
            "No benchmark records loaded."
        )

    # 只处理 attribution CSV 实际覆盖的样本。
    attr_sample_ids = set(
        attribution_df["sample_id"].tolist()
    )

    records = [
        record
        for record in records
        if record["sample_id"] in attr_sample_ids
    ]

    if len(records) == 0:
        raise RuntimeError(
            "No overlap between benchmark samples and attribution CSV."
        )

    print(
        f"Benchmark samples with attribution: {len(records)}"
    )

    print("\nLoading primary model...")

    tokenizer, model = load_model(
        model_name=args.primary_model,
        device=args.device,
        dtype=dtype,
    )

    model_device = get_model_device(model)

    all_result_frames = []
    all_selection_frames = []
    metadata_rows = []

    successful_samples = 0

    for sample_index, record in enumerate(
        records,
        start=1,
    ):
        sample_id = record["sample_id"]

        print("\n" + "#" * 120)
        print(
            f"[{sample_index}/{len(records)}] "
            f"{sample_id}"
        )
        print("#" * 120)

        try:
            (
                result_df,
                selection_df,
                metadata,
            ) = run_one_sample(
                record=record,
                tokenizer=tokenizer,
                model=model,
                device=model_device,
                attribution_df=attribution_df,
                args=args,
            )

            all_result_frames.append(result_df)
            all_selection_frames.append(selection_df)
            metadata_rows.append(metadata)

            successful_samples += 1

            print(
                "  Evidence tokens:",
                metadata["evidence_token_count"],
            )

            print(
                "  Attribution tokens:",
                metadata[
                    "available_attribution_token_count"
                ],
            )

            print(
                "  Full answer:",
                repr(metadata["full_generated_answer"]),
            )

            print(
                "  Reuse answer:",
                repr(metadata["reuse_generated_answer"]),
            )

        except Exception as exc:
            print(
                f"[Warning] Failed sample {sample_id}: {exc}"
            )

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(all_result_frames) == 0:
        raise RuntimeError(
            "No samples completed successfully."
        )

    result_df = pd.concat(
        all_result_frames,
        ignore_index=True,
    )

    selection_df = pd.concat(
        all_selection_frames,
        ignore_index=True,
    )

    metadata_df = pd.DataFrame(
        metadata_rows
    )

    aggregate_df = aggregate_results(
        result_df
    )

    hop_selection_df = aggregate_selection(
        selection_df
    )

    # ========================================================
    # 保存结果
    # ========================================================

    result_path = (
        output_dir
        / "per_sample_results.csv"
    )

    aggregate_path = (
        output_dir
        / "aggregate_results.csv"
    )

    selection_path = (
        output_dir
        / "selected_tokens.csv"
    )

    hop_selection_path = (
        output_dir
        / "hop_selection_summary.csv"
    )

    metadata_csv_path = (
        output_dir
        / "sample_metadata.csv"
    )

    result_df.to_csv(
        result_path,
        index=False,
        encoding="utf-8",
    )

    aggregate_df.to_csv(
        aggregate_path,
        index=False,
        encoding="utf-8",
    )

    selection_df.to_csv(
        selection_path,
        index=False,
        encoding="utf-8",
    )

    hop_selection_df.to_csv(
        hop_selection_path,
        index=False,
        encoding="utf-8",
    )

    metadata_df.to_csv(
        metadata_csv_path,
        index=False,
        encoding="utf-8",
    )

    print(f"[Saved] {result_path}")
    print(f"[Saved] {aggregate_path}")
    print(f"[Saved] {selection_path}")
    print(f"[Saved] {hop_selection_path}")
    print(f"[Saved] {metadata_csv_path}")

# ========================================================
    # 保存运行说明
    # ========================================================

    run_metadata = {
        "benchmark_jsonl": str(
            benchmark_path.resolve()
        ),
        "attribution_csv": str(
            Path(args.attribution_csv).resolve()
        ),
        "output_dir": str(
            output_dir.resolve()
        ),
        "successful_samples": successful_samples,
        "requested_max_samples": args.max_samples,
        "repair_ratios": args.repair_ratios,
        "negative_policy": args.negative_policy,
        "target_generation_step": (
            args.target_generation_step
        ),
        "selector_definition": (
            "Rank prompt evidence tokens by step_kl_reduction "
            "from single-token decode-step causal attribution. "
            "Select top-ratio tokens and patch their K/V "
            "with Target Full Recompute K/V."
        ),
        "is_oracle": True,
        "important_note": (
            "The selector is an ORACLE experiment because step_kl_reduction "
            "is computed using the exact target model outputs. "
            "Use this upper-bound baseline to evaluate heuristic KV repair strategies."
        ),
    }

    run_metadata_path = output_dir / "run_metadata.json"
    with open(run_metadata_path, "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2, ensure_ascii=False)

    print(f"[Saved] {run_metadata_path}")

    # ========================================================
    # 绘制评估图表
    # ========================================================

    print("\nGenerating evaluation figures...")

    try:
        # 1. 绘制 Candidate Accuracy / Contains Gold / Margin 恢复曲线
        plot_accuracy_recovery(
            summary_df=aggregate_df,
            output_dir=figure_dir,
        )

        # 2. 绘制 Next-token KL / Cosine / Top-k Overlap 保真度恢复曲线
        plot_fidelity_recovery(
            summary_df=aggregate_df,
            output_dir=figure_dir,
        )

        # 3. 绘制不同 Repair Ratio 下各 Hop 的 Token 选择分布图
        plot_hop_selection(
            selection_df=hop_selection_df,
            output_dir=figure_dir,
        )

        print(f"[Success] Figures generated in: {figure_dir}")

    except Exception as exc:
        print(f"[Warning] Failed to generate figures: {exc}")

    print("\n" + "=" * 120)
    print("KL-Reduction-Guided Token KV Repair Evaluation Completed Successfully!")
    print("=" * 120)


# ============================================================
# 14. Entry Point
# ============================================================

if __name__ == "__main__":
    main()
