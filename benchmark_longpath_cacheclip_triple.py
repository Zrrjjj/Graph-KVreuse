#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
benchmark_longpath_cacheclip_triple.py

Triple-Level CacheClip KV Repair Benchmark
==========================================

方法：
    1. Full Recompute
    2. Full Long-Path KV Reuse
    3. CacheClip Triple Repair @ multiple requested repair ratios

CacheClip Triple Repair：
    1. 小模型计算 Question -> Evidence Token Attention；
    2. 选出 top-vote-ratio 高注意力 evidence token；
    3. 统计每个 KG triple/hop 中高注意力 token 数量；
    4. 以 high-attention token count 为主要排序依据，
       attention score sum / mean 为 tie-break；
    5. 按完整 triple/hop 选择，直到达到 repair token budget；
    6. 对被选三元组中的所有 token 做 Target-KV Oracle patch；
    7. 从路径后缀重新 Prefill；
    8. 与 Full Recompute 和 Full KV Reuse 对比。

重要：
    本实验的“重计算”是 Oracle KV Refresh：
        Target Full Prefill 中提取 selected triple tokens 的真值 K/V
        -> 覆盖 Source Prompt 中的对应旧 K/V。

    因此这是机制分析 / 精度恢复上界实验，
    不是无需 Target Full Prefill 的部署型加速系统。

输入 JSONL 每条应包含：
    sample_id
    scenario
    hop_length
    source_prompt
    target_prompt
    gold_answers
    target_path
    noise_paths

目标路径格式：
    <KG_PATH id=P_TARGET_LONG block=TARGET_PATH ...>
        <TRIPLE hop=1>...</TRIPLE>
        ...
        <TRIPLE hop=H>...</TRIPLE>
    </KG_PATH>
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

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DynamicCache,
)


# ============================================================
# 1. 基础工具
# ============================================================

def parse_dtype(name):
    name = name.lower()

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
        return cache.to_legacy_cache()

    return cache

def ensure_model_cache(cache):
    """
    Convert legacy tuple KV cache to Transformers DynamicCache
    before passing it into model(...).

    Internal experiment code keeps KV as legacy tuples because
    token/triple-level KV patching requires direct K/V slicing.
    """

    if cache is None:
        return None

    # Already a Transformers Cache / DynamicCache object.
    if hasattr(cache, "get_seq_length"):
        return cache

    # Legacy cache:
    # ((K_layer0, V_layer0), (K_layer1, V_layer1), ...)
    if isinstance(cache, tuple):
        return DynamicCache.from_legacy_cache(cache)

    raise TypeError(
        f"Unsupported cache type: {type(cache)}"
    )


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
    normalized_generated = normalize_text(generated)

    return int(
        any(
            normalize_text(answer) in normalized_generated
            for answer in gold_answers
        )
    )


def extracted_entity_em(generated, gold_answers):
    extracted = normalize_text(
        extract_answer_phrase(generated)
    )

    gold_set = {
        normalize_text(answer)
        for answer in gold_answers
    }

    return int(extracted in gold_set)


def normalize_scores(scores):
    scores = np.asarray(scores, dtype=np.float64)

    if len(scores) == 0:
        return scores

    score_min = scores.min()
    score_max = scores.max()

    if score_max - score_min < 1e-12:
        return np.zeros_like(scores)

    return (scores - score_min) / (
        score_max - score_min
    )


# ============================================================
# 2. Prompt Span 定位
# ============================================================

def find_target_path_char_spans(prompt, hop_length):
    """
    找到目标 KG_PATH、每一个 TRIPLE/hop 的字符范围。
    """

    path_open_pattern = (
        r"<KG_PATH\s+id=P_TARGET_LONG\s+"
        r"block=TARGET_PATH[^>]*>"
    )

    path_open_match = re.search(
        path_open_pattern,
        prompt,
    )

    if path_open_match is None:
        raise RuntimeError(
            "Cannot find target KG_PATH opening tag."
        )

    path_start = path_open_match.start()

    path_close_tag = "</KG_PATH>"

    path_close_position = prompt.find(
        path_close_tag,
        path_open_match.end(),
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
        hop_open_pattern = (
            rf"<TRIPLE\s+hop={hop_id}>"
        )

        hop_open_match = re.search(
            hop_open_pattern,
            prompt[path_start:path_end],
        )

        if hop_open_match is None:
            raise RuntimeError(
                f"Cannot find <TRIPLE hop={hop_id}>."
            )

        hop_start = (
            path_start
            + hop_open_match.start()
        )

        triple_close_tag = "</TRIPLE>"

        hop_close_position = prompt.find(
            triple_close_tag,
            hop_start,
        )

        if hop_close_position < 0:
            raise RuntimeError(
                f"Cannot find </TRIPLE> for hop={hop_id}."
            )

        hop_end = (
            hop_close_position
            + len(triple_close_tag)
        )

        hop_char_spans.append(
            (hop_start, hop_end)
        )

    return {
        "full_path_char_span": (
            path_start,
            path_end,
        ),
        "hop_char_spans": hop_char_spans,
    }


def find_question_char_span(prompt):
    question_marker = "Question: "
    answer_marker = "\nAnswer:"

    question_start = prompt.rfind(
        question_marker
    )

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


# ============================================================
# 3. Tokenization / Offset Mapping
# ============================================================

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
            f"No token overlaps span [{char_start}, {char_end})."
        )

    return positions


def extract_prompt_spans(
    tokenizer,
    prompt,
    hop_length,
):
    """
    evidence_positions:
        所有 triple token 的并集。
        不包含 <KG_PATH> / </KG_PATH> wrapper。

    hop_positions:
        List[List[int]]，每个 triple/hop 的 token positions。

    full_path_token_span:
        连续 KG_PATH block，用于整体复制 Source Path KV。
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

    for hop_start, hop_end in char_info["hop_char_spans"]:
        positions = char_span_to_token_positions(
            offsets,
            hop_start,
            hop_end,
        )

        hop_positions.append(positions)

    evidence_positions = sorted(
        {
            position
            for positions in hop_positions
            for position in positions
        }
    )

    # 所有 triple span 不应该有 token overlap。
    if len(evidence_positions) != sum(
        len(positions)
        for positions in hop_positions
    ):
        raise RuntimeError(
            "Triple/hop token spans overlap unexpectedly."
        )

    full_path_positions = char_span_to_token_positions(
        offsets,
        char_info["full_path_char_span"][0],
        char_info["full_path_char_span"][1],
    )

    full_path_token_span = (
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
            if position in global_to_hop:
                raise RuntimeError(
                    "One evidence token belongs to multiple hops."
                )

            global_to_hop[position] = hop_index

    for position in evidence_positions:
        if position not in global_to_hop:
            raise RuntimeError(
                "Evidence token not assigned to any hop."
            )

    return {
        "input_ids": input_ids,
        "offsets": offsets,
        "evidence_positions": evidence_positions,
        "hop_positions": hop_positions,
        "question_positions": question_positions,
        "full_path_token_span": full_path_token_span,
        "global_to_hop": global_to_hop,
    }


# ============================================================
# 4. 模型加载与 Prefill
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
        attn_implementation="eager",
    )

    model.to(device)
    model.eval()

    return tokenizer, model


@torch.no_grad()
def full_prefill(
    model,
    input_ids_cpu,
    device,
    output_attentions=False,
):
    input_ids = input_ids_cpu.unsqueeze(0).to(device)

    attention_mask = torch.ones_like(input_ids)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=output_attentions,
        use_cache=True,
    )

    result = {
        "input_ids_cpu": input_ids_cpu,
        "seq_len": input_ids.shape[1],
        "logits": outputs.logits[:, -1, :].detach(),
        "cache": to_legacy_cache(
            outputs.past_key_values
        ),
    }

    if output_attentions:
        if outputs.attentions is None:
            raise RuntimeError(
                "No attention returned. Use eager attention."
            )

        result["attentions"] = outputs.attentions

    return result


# ============================================================
# 5. 小模型 CacheClip：Question -> Token Attention
# ============================================================

def resolve_last_k_layers(attentions, last_k):
    total_layers = len(attentions)

    if last_k <= 0:
        raise ValueError(
            "small_attention_last_k must be positive."
        )

    last_k = min(last_k, total_layers)

    return list(
        range(
            total_layers - last_k,
            total_layers,
        )
    )


def calculate_small_token_attention_scores(
    attentions,
    question_positions,
    evidence_positions,
    last_k,
):
    """
    小模型最后 K 层计算：

        Question tokens -> evidence tokens

    单层分数：
        mean over heads
        mean over question tokens

    每层在 evidence token 范围内 min-max normalize，
    最后 K 层平均。
    """

    selected_layers = resolve_last_k_layers(
        attentions,
        last_k,
    )

    normalized_layer_scores = []

    for layer_index in selected_layers:
        # [heads, seq_len, seq_len]
        layer_attention = attentions[layer_index][0].float()

        # [heads, num_question_tokens, num_evidence_tokens]
        q_to_evidence = layer_attention[
            :,
            question_positions,
            :,
        ][:, :, evidence_positions]

        token_scores = q_to_evidence.mean(
            dim=(0, 1)
        ).detach().cpu().numpy()

        normalized_layer_scores.append(
            normalize_scores(token_scores)
        )

    normalized_layer_scores = np.stack(
        normalized_layer_scores,
        axis=0,
    )

    final_scores = normalized_layer_scores.mean(axis=0)

    return {
        "selected_layers": selected_layers,
        "layer_scores": normalized_layer_scores,
        "final_scores": final_scores,
    }


# ============================================================
# 6. 小模型 Token 分数映射至大模型 Token 空间
# ============================================================

def span_overlap_length(span_a, span_b):
    start_a, end_a = span_a
    start_b, end_b = span_b

    return max(
        0,
        min(end_a, end_b) - max(start_a, start_b),
    )


def map_small_scores_to_large_tokens(
    small_scores,
    small_offsets,
    small_evidence_positions,
    large_offsets,
    large_evidence_positions,
):
    """
    小模型与大模型 tokenizer 可能不同。

    根据 token 对应字符 span overlap，
    将小模型 token scores 映射到大模型 evidence token。
    """

    small_token_records = []

    for local_index, global_position in enumerate(
        small_evidence_positions
    ):
        start, end = small_offsets[global_position]

        if start == end:
            continue

        small_token_records.append(
            {
                "span": (start, end),
                "score": float(small_scores[local_index]),
            }
        )

    mapped_scores = []

    for large_global_position in large_evidence_positions:
        large_start, large_end = large_offsets[
            large_global_position
        ]

        weighted_sum = 0.0
        total_weight = 0.0

        for record in small_token_records:
            overlap = span_overlap_length(
                (large_start, large_end),
                record["span"],
            )

            if overlap > 0:
                weighted_sum += overlap * record["score"]
                total_weight += overlap

        if total_weight <= 0:
            mapped_scores.append(0.0)
        else:
            mapped_scores.append(
                weighted_sum / total_weight
            )

    return normalize_scores(
        np.asarray(mapped_scores, dtype=np.float64)
    )


# ============================================================
# 7. Triple-Level CacheClip Selector
# ============================================================

def build_local_index_mapping(evidence_positions):
    """
    global prompt token position -> evidence local index
    """

    return {
        global_position: local_index
        for local_index, global_position in enumerate(
            evidence_positions
        )
    }


def rank_triples_by_high_attention_tokens(
    token_scores,
    evidence_positions,
    hop_positions,
    vote_ratio,
):
    """
    核心 Triple Selector。

    Step 1:
        在所有 evidence token 中选取 top-vote-ratio 的高注意力 token。

    Step 2:
        对每个 triple/hop 统计：
            high_attention_token_count
            attention_score_sum
            attention_score_mean

    Step 3:
        排序优先级：
            1. high_attention_token_count 降序
            2. attention_score_sum 降序
            3. attention_score_mean 降序
            4. hop index 升序（稳定 tie-break）

    返回：
        triple_infos: 按优先级排序的 triple metadata
        high_attention_local_indices
    """

    token_scores = np.asarray(
        token_scores,
        dtype=np.float64,
    )

    evidence_token_count = len(token_scores)

    if evidence_token_count == 0:
        raise RuntimeError(
            "No evidence tokens available."
        )

    vote_budget = max(
        1,
        int(
            math.ceil(
                evidence_token_count * vote_ratio
            )
        ),
    )

    top_token_ranking = np.argsort(token_scores)[::-1]

    high_attention_local_indices = set(
        top_token_ranking[:vote_budget].tolist()
    )

    global_to_local = build_local_index_mapping(
        evidence_positions
    )

    triple_infos = []

    for hop_index, hop_global_positions in enumerate(
        hop_positions,
        start=1,
    ):
        local_indices = []

        for global_position in hop_global_positions:
            if global_position not in global_to_local:
                raise RuntimeError(
                    "Hop token not found in evidence positions."
                )

            local_indices.append(
                global_to_local[global_position]
            )

        triple_scores = token_scores[local_indices]

        high_attention_count = sum(
            int(local_index in high_attention_local_indices)
            for local_index in local_indices
        )

        triple_infos.append(
            {
                "hop_index": hop_index,
                "local_indices": local_indices,
                "token_count": len(local_indices),
                "high_attention_token_count": high_attention_count,
                "attention_score_sum": float(
                    np.sum(triple_scores)
                ),
                "attention_score_mean": float(
                    np.mean(triple_scores)
                ),
                "attention_score_max": float(
                    np.max(triple_scores)
                ),
            }
        )

    triple_infos = sorted(
        triple_infos,
        key=lambda item: (
            -item["high_attention_token_count"],
            -item["attention_score_sum"],
            -item["attention_score_mean"],
            item["hop_index"],
        ),
    )

    return (
        triple_infos,
        sorted(high_attention_local_indices),
    )


def select_triples_by_repair_budget(
    ranked_triples,
    total_evidence_token_count,
    requested_repair_ratio,
):
    """
    按 triple rank 从高到低选择完整 triple，
    直到累计 Triple token 数 >= requested repair budget。

    注意：
        因为基本单位是完整 triple，
        actual repair ratio 通常 >= requested repair ratio。
    """

    if requested_repair_ratio <= 0:
        return [], [], 0, 0.0

    requested_token_budget = max(
        1,
        int(
            math.ceil(
                total_evidence_token_count
                * requested_repair_ratio
            )
        ),
    )

    selected_hops = []
    selected_local_indices = []

    selected_token_count = 0

    for triple in ranked_triples:
        selected_hops.append(
            triple["hop_index"]
        )

        selected_local_indices.extend(
            triple["local_indices"]
        )

        selected_token_count += triple["token_count"]

        if selected_token_count >= requested_token_budget:
            break

    selected_local_indices = sorted(
        set(selected_local_indices)
    )

    actual_ratio = (
        len(selected_local_indices)
        / float(total_evidence_token_count)
    )

    return (
        selected_hops,
        selected_local_indices,
        requested_token_budget,
        actual_ratio,
    )


# ============================================================
# 8. Hybrid KV Cache
# ============================================================

def build_hybrid_path_cache(
    source_cache,
    target_cache,
    source_full_path_span,
    target_full_path_span,
    target_evidence_positions,
    selected_evidence_local_indices,
):
    """
    Hybrid KV：

        [Target Prefix Before KG_PATH]
        +
        [Source Target KG_PATH KV]

    对 selected triples 中全部 token：
        Source KV -> Target true KV

    KG_PATH wrapper token：
        仍然来自 Source KV。

    注意：
        这是 Oracle KV patch，而非真实在线稀疏 forward。
    """

    source_start, source_end = source_full_path_span
    target_start, target_end = target_full_path_span

    source_path_length = source_end - source_start
    target_path_length = target_end - target_start

    if source_path_length != target_path_length:
        raise RuntimeError(
            "Source/Target KG_PATH block token lengths differ."
        )

    selected_set = set(selected_evidence_local_indices)

    hybrid_layers = []

    for (source_k, source_v), (target_k, target_v) in zip(
        source_cache,
        target_cache,
    ):
        # Target 中 path 之前的真实 prefix。
        prefix_k = target_k[:, :, :target_start, :].clone()
        prefix_v = target_v[:, :, :target_start, :].clone()

        # Source 中完整 target path KV block。
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

        # Patch selected triple token。
        for local_index in selected_set:
            target_global_position = target_evidence_positions[
                local_index
            ]

            relative_position = (
                target_global_position - target_start
            )

            if not (
                0 <= relative_position < target_path_length
            ):
                raise RuntimeError(
                    "Selected token is outside target KG_PATH span."
                )

            path_k[:, :, relative_position, :] = target_k[
                :,
                :,
                target_global_position,
                :,
            ]

            path_v[:, :, relative_position, :] = target_v[
                :,
                :,
                target_global_position,
                :,
            ]

        hybrid_k = torch.cat(
            [prefix_k, path_k],
            dim=2,
        )

        hybrid_v = torch.cat(
            [prefix_v, path_v],
            dim=2,
        )

        hybrid_layers.append(
            (hybrid_k, hybrid_v)
        )

    return tuple(hybrid_layers)


@torch.no_grad()
def recompute_target_suffix(
    model,
    hybrid_cache,
    target_input_ids_cpu,
    target_full_path_span,
    device,
):
    """
    从目标路径结束后重新 Prefill：

        后续 path/noise（若存在）
        </KG_SUBGRAPH>
        Question
        Answer:
    """

    _, target_path_end = target_full_path_span

    suffix_ids_cpu = target_input_ids_cpu[
        target_path_end:
    ]

    if len(suffix_ids_cpu) == 0:
        raise RuntimeError(
            "No suffix after target KG_PATH."
        )

    prefix_length = target_path_end

    suffix_ids = suffix_ids_cpu.unsqueeze(0).to(device)

    total_length = prefix_length + suffix_ids.shape[1]

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

    model_cache = ensure_model_cache(hybrid_cache)

    outputs = model(
        input_ids=suffix_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=model_cache,
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
# 9. Generation 与 Candidate Ranking
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

    for step in range(max_new_tokens):
        next_token = torch.argmax(
            logits,
            dim=-1,
            keepdim=True,
        )

        next_token_id = int(next_token.item())

        generated_ids.append(next_token_id)

        if (
            tokenizer.eos_token_id is not None
            and next_token_id == tokenizer.eos_token_id
        ):
            break

        current_position = initial_seq_len + step

        attention_mask = torch.ones(
            (1, current_position + 1),
            dtype=torch.long,
            device=device,
        )

        position_ids = torch.tensor(
            [[current_position]],
            dtype=torch.long,
            device=device,
        )

        model_cache = ensure_model_cache(cache)

        outputs = model(
            input_ids=next_token.to(device),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=model_cache,
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


def build_candidates(record, max_candidates):
    candidates = []
    seen = set()

    for answer in record.get("gold_answers", []):
        if answer not in seen:
            candidates.append(answer)
            seen.add(answer)

    target_path = record.get("target_path", {})

    for node in target_path.get("nodes", []):
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
        candidate_ids = tokenizer.encode(
            " " + candidate,
            add_special_tokens=False,
        )

        if len(candidate_ids) == 0:
            continue

        logits = initial_logits
        cache = initial_cache
        sequence_length = initial_seq_len

        total_log_probability = 0.0

        for token_index, token_id in enumerate(candidate_ids):
            log_probabilities = F.log_softmax(
                logits.float(),
                dim=-1,
            )

            total_log_probability += float(
                log_probabilities[0, token_id].item()
            )

            if token_index == len(candidate_ids) - 1:
                break

            input_token = torch.tensor(
                [[token_id]],
                dtype=torch.long,
                device=device,
            )

            attention_mask = torch.ones(
                (1, sequence_length + 1),
                dtype=torch.long,
                device=device,
            )

            position_ids = torch.tensor(
                [[sequence_length]],
                dtype=torch.long,
                device=device,
            )

            model_cache = ensure_model_cache(cache)

            outputs = model(
                input_ids=input_token,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=model_cache,
                use_cache=True,
            )

            cache = to_legacy_cache(
                outputs.past_key_values
            )

            logits = outputs.logits[:, -1, :]
            sequence_length += 1

        rankings.append(
            {
                "candidate": candidate,
                "total_logprob": total_log_probability,
                "avg_logprob": (
                    total_log_probability
                    / len(candidate_ids)
                ),
            }
        )

    return sorted(
        rankings,
        key=lambda item: item["avg_logprob"],
        reverse=True,
    )


# ============================================================
# 10. Logits Metrics
# ============================================================

def calculate_logits_metrics(
    full_logits,
    method_logits,
    top_k,
):
    full_logits = full_logits.float()
    method_logits = method_logits.float()

    full_probabilities = F.softmax(
        full_logits,
        dim=-1,
    )

    method_log_probabilities = F.log_softmax(
        method_logits,
        dim=-1,
    )

    kl_divergence = F.kl_div(
        method_log_probabilities,
        full_probabilities,
        reduction="batchmean",
    ).item()

    logits_cosine = F.cosine_similarity(
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
        "next_token_kl": kl_divergence,
        "next_token_logits_cosine": logits_cosine,
        "next_token_top1_match": int(
            full_top1 == method_top1
        ),
        "next_token_topk_overlap": (
            len(full_topk & method_topk)
            / float(top_k)
        ),
    }


# ============================================================
# 11. 统一评估行
# ============================================================

def make_result_row(
    record,
    method,
    method_type,
    is_oracle,
    requested_repair_ratio,
    actual_repair_token_ratio,
    selected_triple_count,
    selected_token_count,
    selected_hops,
    full_result,
    method_result,
    generated_answer,
    full_answer,
    rankings,
    top_k,
    analysis_latency,
):
    gold_answers = record.get("gold_answers", [])

    normalized_gold_set = {
        normalize_text(answer)
        for answer in gold_answers
    }

    prediction = (
        rankings[0]["candidate"]
        if rankings
        else ""
    )

    candidate_accuracy = int(
        normalize_text(prediction)
        in normalized_gold_set
    )

    gold_candidate_rank = None

    for rank, item in enumerate(rankings, start=1):
        if (
            normalize_text(item["candidate"])
            in normalized_gold_set
        ):
            gold_candidate_rank = rank
            break

    logits_metrics = calculate_logits_metrics(
        full_logits=full_result["logits"],
        method_logits=method_result["logits"],
        top_k=top_k,
    )

    return {
        "sample_id": record["sample_id"],
        "scenario": record.get("scenario", ""),
        "hop_length": int(record["hop_length"]),

        "method": method,
        "method_type": method_type,
        "is_oracle": int(is_oracle),

        "requested_repair_ratio": requested_repair_ratio,
        "actual_repair_token_ratio": actual_repair_token_ratio,
        "selected_triple_count": selected_triple_count,
        "selected_token_count": selected_token_count,
        "selected_hops": json.dumps(selected_hops),

        "candidate_prediction": prediction,
        "candidate_accuracy": candidate_accuracy,
        "gold_candidate_rank": gold_candidate_rank,

        "generated_answer": generated_answer,
        "contains_gold_answer": contains_gold_answer(
            generated_answer,
            gold_answers,
        ),
        "extracted_entity_em": extracted_entity_em(
            generated_answer,
            gold_answers,
        ),
        "generation_match_full": int(
            normalize_text(generated_answer)
            == normalize_text(full_answer)
        ),

        "candidate_rankings": json.dumps(
            rankings,
            ensure_ascii=False,
        ),

        **logits_metrics,

        "analysis_latency_sec": analysis_latency,
    }


# ============================================================
# 12. 单样本实验
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
    # Source / Target prompt span
    # --------------------------------------------------------
    source_large = extract_prompt_spans(
        large_tokenizer,
        source_prompt,
        hop_length,
    )

    target_large = extract_prompt_spans(
        large_tokenizer,
        target_prompt,
        hop_length,
    )

    target_small = extract_prompt_spans(
        small_tokenizer,
        target_prompt,
        hop_length,
    )

    source_evidence_positions = source_large[
        "evidence_positions"
    ]

    target_evidence_positions = target_large[
        "evidence_positions"
    ]

    if len(source_evidence_positions) != len(
        target_evidence_positions
    ):
        raise RuntimeError(
            "Source/Target evidence token count differs."
        )

    source_evidence_ids = source_large["input_ids"][
        source_evidence_positions
    ]

    target_evidence_ids = target_large["input_ids"][
        target_evidence_positions
    ]

    if not torch.equal(
        source_evidence_ids,
        target_evidence_ids,
    ):
        raise RuntimeError(
            "Source/Target evidence token IDs differ."
        )

    source_full_path_span = source_large[
        "full_path_token_span"
    ]

    target_full_path_span = target_large[
        "full_path_token_span"
    ]

    source_path_start, source_path_end = source_full_path_span
    target_path_start, target_path_end = target_full_path_span

    source_path_ids = source_large["input_ids"][
        source_path_start:source_path_end
    ]

    target_path_ids = target_large["input_ids"][
        target_path_start:target_path_end
    ]

    if not torch.equal(source_path_ids, target_path_ids):
        raise RuntimeError(
            "Source/Target full KG_PATH token IDs differ."
        )

    # --------------------------------------------------------
    # 1. 小模型：Question -> Evidence Token Attention
    # --------------------------------------------------------
    small_result, small_attention_latency = measure_time(
        lambda: full_prefill(
            model=small_model,
            input_ids_cpu=target_small["input_ids"],
            device=small_device,
            output_attentions=True,
        ),
        small_device,
    )

    small_attention_result = (
        calculate_small_token_attention_scores(
            attentions=small_result["attentions"],
            question_positions=target_small[
                "question_positions"
            ],
            evidence_positions=target_small[
                "evidence_positions"
            ],
            last_k=args.small_attention_last_k,
        )
    )

    # 小模型 token 分数 -> 大模型 target evidence token 空间。
    cacheclip_token_scores = (
        map_small_scores_to_large_tokens(
            small_scores=small_attention_result[
                "final_scores"
            ],
            small_offsets=target_small["offsets"],
            small_evidence_positions=target_small[
                "evidence_positions"
            ],
            large_offsets=target_large["offsets"],
            large_evidence_positions=target_large[
                "evidence_positions"
            ],
        )
    )

    # --------------------------------------------------------
    # 2. Triple ranking
    # --------------------------------------------------------
    ranked_triples, high_attention_token_indices = (
        rank_triples_by_high_attention_tokens(
            token_scores=cacheclip_token_scores,
            evidence_positions=target_evidence_positions,
            hop_positions=target_large["hop_positions"],
            vote_ratio=args.triple_vote_ratio,
        )
    )

    # --------------------------------------------------------
    # 3. 大模型 Source / Target Full Prefill
    # --------------------------------------------------------
    source_full, source_prefill_latency = measure_time(
        lambda: full_prefill(
            model=large_model,
            input_ids_cpu=source_large["input_ids"],
            device=large_device,
            output_attentions=False,
        ),
        large_device,
    )

    target_full, target_prefill_latency = measure_time(
        lambda: full_prefill(
            model=large_model,
            input_ids_cpu=target_large["input_ids"],
            device=large_device,
            output_attentions=False,
        ),
        large_device,
    )

    candidates = build_candidates(
        record,
        args.max_candidates,
    )

    # --------------------------------------------------------
    # 4. Full Recompute
    # --------------------------------------------------------
    full_answer, full_decode_latency = measure_time(
        lambda: greedy_generate(
            model=large_model,
            tokenizer=large_tokenizer,
            initial_logits=target_full["logits"],
            initial_cache=target_full["cache"],
            initial_seq_len=target_full["seq_len"],
            device=large_device,
            max_new_tokens=args.max_new_tokens,
        ),
        large_device,
    )

    full_rankings, full_rank_latency = measure_time(
        lambda: rank_candidates(
            model=large_model,
            tokenizer=large_tokenizer,
            initial_logits=target_full["logits"],
            initial_cache=target_full["cache"],
            initial_seq_len=target_full["seq_len"],
            candidates=candidates,
            device=large_device,
        ),
        large_device,
    )

    output_rows = []

    output_rows.append(
        make_result_row(
            record=record,
            method="Full Recompute",
            method_type="full_recompute",
            is_oracle=False,
            requested_repair_ratio=1.0,
            actual_repair_token_ratio=1.0,
            selected_triple_count=hop_length,
            selected_token_count=len(target_evidence_positions),
            selected_hops=list(range(1, hop_length + 1)),
            full_result=target_full,
            method_result=target_full,
            generated_answer=full_answer,
            full_answer=full_answer,
            rankings=full_rankings,
            top_k=args.top_k,
            analysis_latency=(
                target_prefill_latency
                + full_decode_latency
                + full_rank_latency
            ),
        )
    )

    # --------------------------------------------------------
    # 5. Full Long-Path KV Reuse / 0% repair
    # --------------------------------------------------------
    reuse_cache, reuse_patch_latency = measure_time(
        lambda: build_hybrid_path_cache(
            source_cache=source_full["cache"],
            target_cache=target_full["cache"],
            source_full_path_span=source_full_path_span,
            target_full_path_span=target_full_path_span,
            target_evidence_positions=target_evidence_positions,
            selected_evidence_local_indices=[],
        ),
        large_device,
    )

    reuse_result, reuse_suffix_latency = measure_time(
        lambda: recompute_target_suffix(
            model=large_model,
            hybrid_cache=reuse_cache,
            target_input_ids_cpu=target_large["input_ids"],
            target_full_path_span=target_full_path_span,
            device=large_device,
        ),
        large_device,
    )

    reuse_answer, reuse_decode_latency = measure_time(
        lambda: greedy_generate(
            model=large_model,
            tokenizer=large_tokenizer,
            initial_logits=reuse_result["logits"],
            initial_cache=reuse_result["cache"],
            initial_seq_len=reuse_result["seq_len"],
            device=large_device,
            max_new_tokens=args.max_new_tokens,
        ),
        large_device,
    )

    reuse_rankings, reuse_rank_latency = measure_time(
        lambda: rank_candidates(
            model=large_model,
            tokenizer=large_tokenizer,
            initial_logits=reuse_result["logits"],
            initial_cache=reuse_result["cache"],
            initial_seq_len=reuse_result["seq_len"],
            candidates=candidates,
            device=large_device,
        ),
        large_device,
    )

    output_rows.append(
        make_result_row(
            record=record,
            method="Full Long-Path KV Reuse",
            method_type="full_reuse",
            is_oracle=False,
            requested_repair_ratio=0.0,
            actual_repair_token_ratio=0.0,
            selected_triple_count=0,
            selected_token_count=0,
            selected_hops=[],
            full_result=target_full,
            method_result=reuse_result,
            generated_answer=reuse_answer,
            full_answer=full_answer,
            rankings=reuse_rankings,
            top_k=args.top_k,
            analysis_latency=(
                source_prefill_latency
                + target_prefill_latency
                + small_attention_latency
                + reuse_patch_latency
                + reuse_suffix_latency
                + reuse_decode_latency
                + reuse_rank_latency
            ),
        )
    )

    # --------------------------------------------------------
    # 6. CacheClip Triple Repair at requested ratios
    # --------------------------------------------------------
    token_rows = []
    triple_rows = []

    target_token_ids = target_large["input_ids"].tolist()

    for requested_ratio in args.repair_ratios:
        (
            selected_hops,
            selected_local_indices,
            requested_token_budget,
            actual_token_ratio,
        ) = select_triples_by_repair_budget(
            ranked_triples=ranked_triples,
            total_evidence_token_count=len(
                target_evidence_positions
            ),
            requested_repair_ratio=requested_ratio,
        )

        hybrid_cache, patch_latency = measure_time(
            lambda: build_hybrid_path_cache(
                source_cache=source_full["cache"],
                target_cache=target_full["cache"],
                source_full_path_span=source_full_path_span,
                target_full_path_span=target_full_path_span,
                target_evidence_positions=target_evidence_positions,
                selected_evidence_local_indices=selected_local_indices,
            ),
            large_device,
        )

        triple_result, suffix_latency = measure_time(
            lambda: recompute_target_suffix(
                model=large_model,
                hybrid_cache=hybrid_cache,
                target_input_ids_cpu=target_large["input_ids"],
                target_full_path_span=target_full_path_span,
                device=large_device,
            ),
            large_device,
        )

        generated_answer, decode_latency = measure_time(
            lambda: greedy_generate(
                model=large_model,
                tokenizer=large_tokenizer,
                initial_logits=triple_result["logits"],
                initial_cache=triple_result["cache"],
                initial_seq_len=triple_result["seq_len"],
                device=large_device,
                max_new_tokens=args.max_new_tokens,
            ),
            large_device,
        )

        rankings, ranking_latency = measure_time(
            lambda: rank_candidates(
                model=large_model,
                tokenizer=large_tokenizer,
                initial_logits=triple_result["logits"],
                initial_cache=triple_result["cache"],
                initial_seq_len=triple_result["seq_len"],
                candidates=candidates,
                device=large_device,
            ),
            large_device,
        )

        output_rows.append(
            make_result_row(
                record=record,
                method=(
                    f"CacheClip Triple Repair "
                    f"{requested_ratio:.0%}"
                ),
                method_type="cacheclip_triple_oracle_patch",
                is_oracle=True,
                requested_repair_ratio=requested_ratio,
                actual_repair_token_ratio=actual_token_ratio,
                selected_triple_count=len(selected_hops),
                selected_token_count=len(selected_local_indices),
                selected_hops=selected_hops,
                full_result=target_full,
                method_result=triple_result,
                generated_answer=generated_answer,
                full_answer=full_answer,
                rankings=rankings,
                top_k=args.top_k,
                analysis_latency=(
                    source_prefill_latency
                    + target_prefill_latency
                    + small_attention_latency
                    + patch_latency
                    + suffix_latency
                    + decode_latency
                    + ranking_latency
                ),
            )
        )

        selected_token_set = set(selected_local_indices)
        selected_hop_set = set(selected_hops)

        # Token-level selection log。
        for local_index, global_position in enumerate(
            target_evidence_positions
        ):
            token_id = int(target_token_ids[global_position])

            token_label = large_tokenizer.convert_ids_to_tokens(
                token_id
            )

            if token_label is None:
                token_label = str(token_id)

            hop_index = target_large["global_to_hop"][
                global_position
            ]

            token_rows.append(
                {
                    "sample_id": sample_id,
                    "requested_repair_ratio": requested_ratio,
                    "actual_repair_token_ratio": actual_token_ratio,
                    "path_local_index": local_index,
                    "global_token_position": global_position,
                    "token_id": token_id,
                    "token_label": token_label.replace("\n", "\\n"),
                    "hop_index": hop_index,
                    "cacheclip_token_score": float(
                        cacheclip_token_scores[local_index]
                    ),
                    "is_high_attention_vote_token": int(
                        local_index in high_attention_token_indices
                    ),
                    "selected_for_triple_repair": int(
                        local_index in selected_token_set
                    ),
                    "selected_hop_for_repair": int(
                        hop_index in selected_hop_set
                    ),
                }
            )

        # Triple ranking log。
        for rank, triple_info in enumerate(
            ranked_triples,
            start=1,
        ):
            triple_rows.append(
                {
                    "sample_id": sample_id,
                    "requested_repair_ratio": requested_ratio,
                    "actual_repair_token_ratio": actual_token_ratio,
                    "triple_rank": rank,
                    "hop_index": triple_info["hop_index"],
                    "triple_token_count": triple_info["token_count"],
                    "high_attention_token_count": (
                        triple_info["high_attention_token_count"]
                    ),
                    "attention_score_sum": (
                        triple_info["attention_score_sum"]
                    ),
                    "attention_score_mean": (
                        triple_info["attention_score_mean"]
                    ),
                    "attention_score_max": (
                        triple_info["attention_score_max"]
                    ),
                    "selected_for_repair": int(
                        triple_info["hop_index"] in selected_hop_set
                    ),
                }
            )

        del hybrid_cache
        del triple_result

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metadata = {
        "sample_id": sample_id,
        "small_attention_layers": small_attention_result[
            "selected_layers"
        ],
        "triple_vote_ratio": args.triple_vote_ratio,
        "high_attention_vote_token_count": len(
            high_attention_token_indices
        ),
        "evidence_token_count": len(target_evidence_positions),
        "hop_length": hop_length,
        "source_path_start_token": source_path_start,
        "target_path_start_token": target_path_start,
        "path_position_shift_tokens": (
            target_path_start - source_path_start
        ),
        "ranked_hops": [
            triple["hop_index"]
            for triple in ranked_triples
        ],
    }

    del small_result
    del source_full
    del target_full

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        output_rows,
        pd.DataFrame(token_rows),
        pd.DataFrame(triple_rows),
        metadata,
    )


# ============================================================
# 13. 聚合
# ============================================================

def aggregate_results(result_df):
    grouped = (
        result_df.groupby(
            [
                "method",
                "method_type",
                "is_oracle",
                "requested_repair_ratio",
            ],
            as_index=False,
        )
        .agg(
            n_samples=("sample_id", "nunique"),

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
            mean_gold_candidate_rank=(
                "gold_candidate_rank",
                "mean",
            ),

            mean_kl=(
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

            mean_actual_repair_token_ratio=(
                "actual_repair_token_ratio",
                "mean",
            ),
            mean_selected_triple_count=(
                "selected_triple_count",
                "mean",
            ),
            mean_selected_token_count=(
                "selected_token_count",
                "mean",
            ),

            mean_analysis_latency_sec=(
                "analysis_latency_sec",
                "mean",
            ),
        )
    )

    def method_order(row):
        if row["method_type"] == "full_reuse":
            return 0.0

        if row["method_type"] == "cacheclip_triple_oracle_patch":
            return 1.0 + float(
                row["requested_repair_ratio"]
            )

        if row["method_type"] == "full_recompute":
            return 3.0

        return 99.0

    grouped["method_order"] = grouped.apply(
        method_order,
        axis=1,
    )

    grouped = grouped.sort_values(
        "method_order"
    ).reset_index(drop=True)

    # Direct -> Full recovery。
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
        "mean_topk_overlap",
    ]

    if len(reuse_rows) > 0 and len(full_rows) > 0:
        reuse_row = reuse_rows.iloc[0]
        full_row = full_rows.iloc[0]

        for metric in metrics:
            reuse_value = float(reuse_row[metric])
            full_value = float(full_row[metric])

            denominator = full_value - reuse_value

            recovery_column = f"{metric}_recovery"

            if abs(denominator) < 1e-12:
                grouped[recovery_column] = np.nan
            else:
                grouped[recovery_column] = (
                    grouped[metric] - reuse_value
                ) / denominator

    return grouped


def aggregate_triple_selection(triple_df):
    if len(triple_df) == 0:
        return pd.DataFrame()

    return (
        triple_df.groupby(
            [
                "requested_repair_ratio",
                "hop_index",
            ],
            as_index=False,
        )
        .agg(
            n_samples=("sample_id", "nunique"),
            selection_rate=(
                "selected_for_repair",
                "mean",
            ),
            mean_rank=(
                "triple_rank",
                "mean",
            ),
            mean_high_attention_token_count=(
                "high_attention_token_count",
                "mean",
            ),
            mean_attention_score_sum=(
                "attention_score_sum",
                "mean",
            ),
            mean_attention_score_mean=(
                "attention_score_mean",
                "mean",
            ),
        )
    )


# ============================================================
# 14. JSONL Loader
# ============================================================

def load_jsonl(path, max_samples):
    records = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

            if len(records) >= max_samples:
                break

    return records


# ============================================================
# 15. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Triple-level CacheClip Oracle KV repair benchmark."
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
        help=(
            "Use final K auxiliary-model layers. "
            "1 best matches the original CacheClip-style setup; "
            "4 uses a more stable last-4-layer mean."
        ),
    )

    parser.add_argument(
        "--triple_vote_ratio",
        type=float,
        default=0.20,
        help=(
            "First select top vote_ratio high-attention tokens; "
            "then count these tokens in each triple to rank triples."
        ),
    )

    parser.add_argument(
        "--repair_ratios",
        type=float,
        nargs="+",
        default=[
            0.15,
            0.25,
            0.35,
            0.45,
            0.50,
            0.60,
            0.80,
        ],
        help=(
            "Requested triple-level repair token budgets. "
            "Actual ratios may be larger because complete triples are selected."
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

    if not (
        0 < args.triple_vote_ratio <= 1
    ):
        raise ValueError(
            "triple_vote_ratio must be in (0, 1]."
        )

    for ratio in args.repair_ratios:
        if not (0 < ratio < 1):
            raise ValueError(
                "repair_ratios must be strictly between 0 and 1. "
                "Full Reuse and Full Recompute are included automatically."
            )

    benchmark_path = Path(args.benchmark_jsonl)
    output_dir = Path(args.output_dir)

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
            "CUDA requested but unavailable."
        )

    dtype = parse_dtype(args.dtype)

    print("=" * 120)
    print("Triple-Level CacheClip KV Repair Benchmark")
    print("=" * 120)
    print(f"Benchmark: {benchmark_path}")
    print(f"Output: {output_dir}")
    print(f"Primary Model: {args.primary_model}")
    print(f"Auxiliary Model: {args.auxiliary_model}")
    print(f"Small Attention Last-K: {args.small_attention_last_k}")
    print(f"Triple Vote Ratio: {args.triple_vote_ratio:.2%}")
    print(f"Requested Repair Ratios: {args.repair_ratios}")
    print("=" * 120)

    records = load_jsonl(
        benchmark_path,
        args.max_samples,
    )

    if len(records) == 0:
        raise RuntimeError("No records loaded.")

    print(f"Loaded samples: {len(records)}")

    print("\nLoading auxiliary model...")

    small_tokenizer, small_model = load_model(
        args.auxiliary_model,
        args.device,
        dtype,
    )

    print("Loading primary model...")

    large_tokenizer, large_model = load_model(
        args.primary_model,
        args.device,
        dtype,
    )

    small_device = get_model_device(small_model)
    large_device = get_model_device(large_model)

    all_result_rows = []
    all_token_frames = []
    all_triple_frames = []
    all_metadata = []

    successful_samples = 0

    for sample_index, record in enumerate(records, start=1):
        sample_id = record.get(
            "sample_id",
            f"sample_{sample_index:05d}",
        )

        print(
            f"\n[{sample_index}/{len(records)}] {sample_id}"
        )

        try:
            (
                result_rows,
                token_frame,
                triple_frame,
                metadata,
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

            all_result_rows.extend(result_rows)
            all_token_frames.append(token_frame)
            all_triple_frames.append(triple_frame)
            all_metadata.append(metadata)

            successful_samples += 1

            print(
                f"  Small attention layers: "
                f"{metadata['small_attention_layers']}"
            )

            print(
                f"  High-attention vote tokens: "
                f"{metadata['high_attention_vote_token_count']}"
                f"/{metadata['evidence_token_count']}"
            )

            print(
                f"  Triple ranking: {metadata['ranked_hops']}"
            )

        except Exception as exception:
            print(
                f"[Warning] Failed sample {sample_id}: "
                f"{exception}"
            )

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(all_result_rows) == 0:
        raise RuntimeError(
            "No sample completed successfully."
        )

    result_df = pd.DataFrame(all_result_rows)

    token_df = pd.concat(
        all_token_frames,
        ignore_index=True,
    )

    triple_df = pd.concat(
        all_triple_frames,
        ignore_index=True,
    )

    aggregate_df = aggregate_results(result_df)

    triple_selection_df = aggregate_triple_selection(
        triple_df
    )

    result_df.to_csv(
        output_dir / "per_sample_results.csv",
        index=False,
        encoding="utf-8",
    )

    aggregate_df.to_csv(
        output_dir / "aggregate_results.csv",
        index=False,
        encoding="utf-8",
    )

    token_df.to_csv(
        output_dir / "token_selection.csv",
        index=False,
        encoding="utf-8",
    )

    triple_df.to_csv(
        output_dir / "triple_ranking.csv",
        index=False,
        encoding="utf-8",
    )

    triple_selection_df.to_csv(
        output_dir / "triple_selection_summary.csv",
        index=False,
        encoding="utf-8",
    )

    (output_dir / "sample_metadata.json").write_text(
        json.dumps(
            all_metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    (output_dir / "run_config.json").write_text(
        json.dumps(
            vars(args),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 120)
    print("Aggregate Results")
    print("=" * 120)

    print(
        aggregate_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\n" + "=" * 120)
    print("Triple Selection Summary")
    print("=" * 120)

    print(
        triple_selection_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\n" + "=" * 120)
    print(
        f"Completed: {successful_samples}/{len(records)} samples"
    )
    print(f"Output: {output_dir.resolve()}")
    print("=" * 120)


if __name__ == "__main__":
    main()
