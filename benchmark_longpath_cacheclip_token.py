#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
benchmark_longpath_cacheclip_token.py

Token-Level CacheClip KV Repair on MetaQA Long-Path Benchmark
==============================================================

对比方法：
    1. Full Recompute
    2. Full Long-Path KV Reuse
    3. CacheClip Token Repair @ 10%, 20%, 30%, 40%

CacheClip Token Repair：
    Small Model:
        Question Tokens -> Target KG Evidence Tokens attention
        -> select top-ratio high-attention tokens

    Large Model:
        Target Prefix KV
        + Source Full Target-Path KV
        + patch selected evidence tokens with Target true KV
        + recompute Target suffix

重要：
    - 选择单位：单个 tokenizer token；
    - Patch 单位：单个 tokenizer token；
    - Evidence 范围：仅 <TRIPLE hop=i>...</TRIPLE> 内 token；
    - 不选择 <KG_PATH> / </KG_PATH> wrapper token；
    - 小模型和大模型 tokenizer 不同时，通过字符 span 对齐；
    - Target Full KV patch 是 Oracle 分析，不是线上真实可部署重计算。
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

from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    DynamicCache = None


# ============================================================
# 基础工具
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
    elapsed = time.perf_counter() - start

    return result, elapsed


def to_legacy_cache(cache):
    if hasattr(cache, "to_legacy_cache"):
        return cache


def to_model_cache(cache):
    """
    将手动拼接的 legacy KV tuple 转换为新版 Transformers
    forward 所需的 DynamicCache。

    legacy cache 格式：
        tuple[(K_layer0, V_layer0), ..., (K_layerN, V_layerN)]

    新版 Transformers 中，Qwen forward 会调用：
        past_key_values.get_seq_length()

    因此不能直接传 legacy tuple。
    """

    if cache is None:
        return None

    # 已经是新 Cache 对象，例如 DynamicCache / StaticCache。
    if hasattr(cache, "get_seq_length"):
        return cache

    # 手动 KV 拼接后得到的是 legacy tuple。
    if isinstance(cache, tuple):
        if DynamicCache is None:
            raise RuntimeError(
                "DynamicCache cannot be imported. "
                "Please upgrade transformers, e.g. "
                "`pip install -U transformers`."
            )

        return DynamicCache.from_legacy_cache(cache)

    raise TypeError(
        f"Unsupported cache type: {type(cache)}"
    ).to_legacy_cache()

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
    generated = normalize_text(generated)

    return int(
        any(
            normalize_text(answer) in generated
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


def normalize_scores(scores):
    scores = np.asarray(scores, dtype=np.float64)

    if len(scores) == 0:
        return scores

    min_value = scores.min()
    max_value = scores.max()

    if max_value - min_value < 1e-12:
        return np.zeros_like(scores)

    return (scores - min_value) / (
        max_value - min_value
    )


# ============================================================
# Prompt Span 提取
# ============================================================

def find_target_path_char_spans(prompt, hop_length):
    """
    找到：
        <KG_PATH id=P_TARGET_LONG block=TARGET_PATH ...>
            <TRIPLE hop=1>...</TRIPLE>
            ...
        </KG_PATH>
    """

    open_pattern = (
        r"<KG_PATH\s+id=P_TARGET_LONG\s+"
        r"block=TARGET_PATH[^>]*>"
    )

    open_match = re.search(open_pattern, prompt)

    if open_match is None:
        raise RuntimeError(
            "Cannot find target path opening tag."
        )

    path_start = open_match.start()

    close_tag = "</KG_PATH>"

    close_position = prompt.find(
        close_tag,
        open_match.end(),
    )

    if close_position < 0:
        raise RuntimeError(
            "Cannot find target path closing tag."
        )

    path_end = close_position + len(close_tag)

    hop_char_spans = []

    for hop_id in range(1, hop_length + 1):
        hop_pattern = rf"<TRIPLE\s+hop={hop_id}>"

        hop_match = re.search(
            hop_pattern,
            prompt[path_start:path_end],
        )

        if hop_match is None:
            raise RuntimeError(
                f"Cannot find <TRIPLE hop={hop_id}>."
            )

        hop_start = path_start + hop_match.start()

        triple_close = "</TRIPLE>"

        hop_end_position = prompt.find(
            triple_close,
            hop_start,
        )

        if hop_end_position < 0:
            raise RuntimeError(
                f"Cannot find closing tag for hop={hop_id}."
            )

        hop_end = hop_end_position + len(triple_close)

        hop_char_spans.append((hop_start, hop_end))

    return {
        "full_path_char_span": (path_start, path_end),
        "hop_char_spans": hop_char_spans,
    }


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
    mode="overlap",
):
    """
    将字符 span 映射到 tokenizer token positions。

    mode:
        overlap:
            token 与字符区间有任意 overlap 时选中。
            适用于 Question 等自然语言区域。

        strict:
            token 必须完整位于 [char_start, char_end) 内。
            适用于可复用的 KG_PATH / TRIPLE evidence span。

    为什么 KG_PATH 使用 strict：
        tokenizer 可能产生跨越 </KG_PATH> 边界的 token。
        Source 与 Target 的 KG_PATH 后续文本不同，因此此类
        边界 token 不能被纳入可复用 KV block。
    """

    positions = []

    for token_index, (start, end) in enumerate(offsets):
        # Special token / empty offset。
        if start == end:
            continue

        if mode == "overlap":
            if end > char_start and start < char_end:
                positions.append(token_index)

        elif mode == "strict":
            if start >= char_start and end <= char_end:
                positions.append(token_index)

        else:
            raise ValueError(
                f"Unsupported token span mode: {mode}"
            )

    if len(positions) == 0:
        raise RuntimeError(
            f"No token found for char span "
            f"[{char_start}, {char_end}) "
            f"with mode={mode}."
        )

    return positions

def extract_prompt_spans(
    tokenizer,
    prompt,
    hop_length,
):
    """
    evidence_positions：
        只包含 <TRIPLE hop=i>...</TRIPLE> 中的 token。

    full_path_token_span：
        包含完整 <KG_PATH> ... </KG_PATH> token block。
        用于复制连续 path KV。
    """

    tokenized = tokenize_with_offsets(tokenizer, prompt)

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
            mode="strict",
        )
        hop_positions.append(positions)

    evidence_positions = sorted(
        {
            position
            for positions in hop_positions
            for position in positions
        }
    )

    # 每个 token 必须只属于一个 hop。
    if len(evidence_positions) != sum(
        len(positions)
        for positions in hop_positions
    ):
        raise RuntimeError(
            "Hop token spans overlap unexpectedly."
        )

    full_path_positions = char_span_to_token_positions(
        offsets,
        char_info["full_path_char_span"][0],
        char_info["full_path_char_span"][1],
        mode="strict",
    )

    # Strict reusable KG_PATH token span contiguity check.
    # Boundary-crossing tokenizer tokens are intentionally excluded.
    expected_full_path_positions = list(
        range(
            full_path_positions[0],
            full_path_positions[-1] + 1,
        )
    )

    if full_path_positions != expected_full_path_positions:
        raise RuntimeError(
            "Strict KG_PATH token positions are not contiguous. "
            "Tokenizer boundary crossing exists inside the reusable path span."
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
        mode="overlap",
    )

    global_to_hop = {}

    for hop_index, positions in enumerate(
        hop_positions,
        start=1,
    ):
        for position in positions:
            global_to_hop[position] = hop_index

    for position in evidence_positions:
        if position not in global_to_hop:
            raise RuntimeError(
                "Evidence token not assigned to a hop."
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
# 模型与 Full Prefill
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
        dtype=dtype,
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
        "cache": to_legacy_cache(outputs.past_key_values),
    }

    if output_attentions:
        if outputs.attentions is None:
            raise RuntimeError(
                "No attention returned. Use eager attention."
            )

        result["attentions"] = outputs.attentions

    return result


# ============================================================
# 小模型 CacheClip Token Selector
# ============================================================

def resolve_last_k_layers(attentions, last_k):
    total_layers = len(attentions)

    if last_k <= 0:
        raise ValueError("small_attention_last_k must be positive.")

    last_k = min(last_k, total_layers)

    return list(
        range(
            total_layers - last_k,
            total_layers,
        )
    )


def calculate_small_attention_scores(
    attentions,
    question_positions,
    evidence_positions,
    last_k,
):
    """
    对小模型最后 K 层，计算：

        Question Token -> Evidence Token Attention

    单层公式：

        score(l, t)
        = mean_heads(mean_question_tokens(A[l, h, q, t]))

    每层在 evidence token 范围内归一化，
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

        # [heads, question_token_count, evidence_token_count]
        q_to_evidence = layer_attention[
            :,
            question_positions,
            :,
        ][:, :, evidence_positions]

        # [evidence_token_count]
        scores = q_to_evidence.mean(
            dim=(0, 1)
        ).detach().cpu().numpy()

        normalized_layer_scores.append(
            normalize_scores(scores)
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


def span_overlap_length(a, b):
    a_start, a_end = a
    b_start, b_end = b

    return max(
        0,
        min(a_end, b_end) - max(a_start, b_start),
    )


def map_small_scores_to_large_tokens(
    small_scores,
    small_offsets,
    small_evidence_positions,
    large_offsets,
    large_evidence_positions,
):
    """
    小模型和大模型 tokenizer 可能不同。

    本函数按照字符 span overlap，将：
        small evidence token score
    映射为：
        large evidence token score。
    """

    small_records = []

    for local_index, global_position in enumerate(
        small_evidence_positions
    ):
        start, end = small_offsets[global_position]

        if start == end:
            continue

        small_records.append(
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

        for record in small_records:
            weight = span_overlap_length(
                (large_start, large_end),
                record["span"],
            )

            if weight > 0:
                weighted_sum += weight * record["score"]
                total_weight += weight

        if total_weight <= 0:
            mapped_scores.append(0.0)
        else:
            mapped_scores.append(
                weighted_sum / total_weight
            )

    return normalize_scores(
        np.asarray(mapped_scores, dtype=np.float64)
    )


def select_top_ratio_tokens(scores, ratio):
    scores = np.asarray(scores, dtype=np.float64)

    token_count = len(scores)

    if ratio <= 0:
        return [], 0

    if ratio >= 1:
        return list(range(token_count)), token_count

    budget = max(
        1,
        int(math.ceil(token_count * ratio)),
    )

    ranking = np.argsort(scores)[::-1]

    selected = sorted(ranking[:budget].tolist())

    return selected, budget


# ============================================================
# Hybrid KV Cache：Target Prefix + Source Path + Token Patch
# ============================================================

def build_hybrid_path_cache(
    source_cache,
    target_cache,
    source_full_path_span,
    target_full_path_span,
    target_evidence_positions,
    selected_local_indices,
):
    """
    构造 hybrid KV：

        [Target Prefix Before KG_PATH]
        +
        [Source KG_PATH KV]

    对 selected_local_indices 对应 evidence token：
        Source KV -> Target true KV

    Wrapper token，例如 <KG_PATH> / </KG_PATH>：
        保留 Source KV。

    所有未选中 TRIPLE token：
        保留 Source KV。
    """

    source_start, source_end = source_full_path_span
    target_start, target_end = target_full_path_span

    source_path_len = source_end - source_start
    target_path_len = target_end - target_start

    if source_path_len != target_path_len:
        raise RuntimeError(
            "Source/Target full KG_PATH token lengths differ."
        )

    selected_set = set(selected_local_indices)

    hybrid_layers = []

    for (source_k, source_v), (target_k, target_v) in zip(
        source_cache,
        target_cache,
    ):
        # 真实 Target Prefix KV。
        prefix_k = target_k[:, :, :target_start, :].clone()
        prefix_v = target_v[:, :, :target_start, :].clone()

        # Source 中完整目标 KG_PATH KV。
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

        # 对高注意力 token 做 Target KV patch。
        for local_index in selected_set:
            target_global_position = target_evidence_positions[
                local_index
            ]

            relative_position = (
                target_global_position - target_start
            )

            if not 0 <= relative_position < target_path_len:
                raise RuntimeError(
                    "Selected evidence token lies outside KG_PATH."
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

        hybrid_k = torch.cat([prefix_k, path_k], dim=2)
        hybrid_v = torch.cat([prefix_v, path_v], dim=2)

        hybrid_layers.append((hybrid_k, hybrid_v))

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
    从 KG_PATH 结束位置重新计算 Target suffix：

        后续 Noise Path（如果有）
        </KG_SUBGRAPH>
        Question
        Answer:
    """

    _, path_end = target_full_path_span

    suffix_ids_cpu = target_input_ids_cpu[path_end:]

    if len(suffix_ids_cpu) == 0:
        raise RuntimeError("Target suffix is empty.")

    prefix_len = path_end

    suffix_ids = suffix_ids_cpu.unsqueeze(0).to(device)

    total_len = prefix_len + suffix_ids.shape[1]

    attention_mask = torch.ones(
        (1, total_len),
        dtype=torch.long,
        device=device,
    )

    position_ids = torch.arange(
        prefix_len,
        total_len,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    outputs = model(
        input_ids=suffix_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=to_model_cache(hybrid_cache),
        use_cache=True,
    )

    return {
        "logits": outputs.logits[:, -1, :].detach(),
        "cache": to_legacy_cache(outputs.past_key_values),
        "seq_len": total_len,
    }


# ============================================================
# Generation / Candidate Ranking
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

        outputs = model(
            input_ids=next_token.to(device),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=to_model_cache(cache),
            use_cache=True,
        )

        cache = to_legacy_cache(outputs.past_key_values)
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

    for node in record.get("target_path", {}).get("nodes", []):
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
        seq_len = initial_seq_len

        total_logprob = 0.0

        for token_index, token_id in enumerate(candidate_ids):
            log_probs = F.log_softmax(
                logits.float(),
                dim=-1,
            )

            total_logprob += float(log_probs[0, token_id].item())

            if token_index == len(candidate_ids) - 1:
                break

            token_tensor = torch.tensor(
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
                input_ids=token_tensor,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=to_model_cache(cache),
                use_cache=True,
            )

            cache = to_legacy_cache(outputs.past_key_values)
            logits = outputs.logits[:, -1, :]
            seq_len += 1

        rankings.append(
            {
                "candidate": candidate,
                "total_logprob": total_logprob,
                "avg_logprob": total_logprob / len(candidate_ids),
            }
        )

    return sorted(
        rankings,
        key=lambda x: x["avg_logprob"],
        reverse=True,
    )


# ============================================================
# Logits Metrics
# ============================================================

def calculate_logits_metrics(
    full_logits,
    method_logits,
    top_k,
):
    full_logits = full_logits.float()
    method_logits = method_logits.float()

    full_probs = F.softmax(full_logits, dim=-1)
    method_log_probs = F.log_softmax(method_logits, dim=-1)

    kl = F.kl_div(
        method_log_probs,
        full_probs,
        reduction="batchmean",
    ).item()

    cosine = F.cosine_similarity(
        full_logits,
        method_logits,
        dim=-1,
    ).mean().item()

    full_top1 = int(torch.argmax(full_logits, dim=-1).item())
    method_top1 = int(torch.argmax(method_logits, dim=-1).item())

    full_topk = set(
        torch.topk(full_logits, k=top_k, dim=-1)
        .indices[0]
        .tolist()
    )

    method_topk = set(
        torch.topk(method_logits, k=top_k, dim=-1)
        .indices[0]
        .tolist()
    )

    return {
        "mean_kl": kl,
        "mean_logits_cosine": cosine,
        "mean_top1_match": int(full_top1 == method_top1),
        "mean_topk_overlap": len(full_topk & method_topk) / float(top_k),
    }


# ============================================================
# 单方法评估
# ============================================================

def make_result_row(
    record,
    method,
    method_type,
    is_oracle,
    repair_ratio,
    selected_token_count,
    full_result,
    method_result,
    generated_answer,
    full_answer,
    rankings,
    top_k,
    latency,
):
    gold_answers = record.get("gold_answers", [])

    gold_set = {
        normalize_text(answer)
        for answer in gold_answers
    }

    prediction = rankings[0]["candidate"] if rankings else ""

    candidate_accuracy = int(
        normalize_text(prediction) in gold_set
    )

    gold_rank = None

    for rank, item in enumerate(rankings, start=1):
        if normalize_text(item["candidate"]) in gold_set:
            gold_rank = rank
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

        "repair_ratio": repair_ratio,
        "selected_token_count": selected_token_count,

        "candidate_prediction": prediction,
        "candidate_accuracy": candidate_accuracy,
        "gold_candidate_rank": gold_rank,

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

        "analysis_latency_sec": latency,
    }


# ============================================================
# 单样本实验
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
    # 解析 Source / Target Prompt Span
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

    source_evidence_positions = source_large["evidence_positions"]
    target_evidence_positions = target_large["evidence_positions"]

    if len(source_evidence_positions) != len(target_evidence_positions):
        raise RuntimeError(
            "Source/Target evidence token counts differ."
        )

    source_evidence_ids = source_large["input_ids"][
        source_evidence_positions
    ]

    target_evidence_ids = target_large["input_ids"][
        target_evidence_positions
    ]

    if not torch.equal(source_evidence_ids, target_evidence_ids):
        raise RuntimeError(
            "Source/Target evidence token IDs differ."
        )

    source_full_path_span = source_large["full_path_token_span"]
    target_full_path_span = target_large["full_path_token_span"]

    source_path_start, source_path_end = source_full_path_span
    target_path_start, target_path_end = target_full_path_span

    source_full_path_ids = source_large["input_ids"][
        source_path_start:source_path_end
    ]

    target_full_path_ids = target_large["input_ids"][
        target_path_start:target_path_end
    ]

    if not torch.equal(source_full_path_ids, target_full_path_ids):
        raise RuntimeError(
            "Source/Target full KG_PATH token IDs differ."
        )

    # --------------------------------------------------------
    # 1. 小模型：Query -> Evidence Token Attention
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

    small_attention_result = calculate_small_attention_scores(
        attentions=small_result["attentions"],
        question_positions=target_small["question_positions"],
        evidence_positions=target_small["evidence_positions"],
        last_k=args.small_attention_last_k,
    )

    small_raw_scores = small_attention_result["final_scores"]

    # 对齐到大模型 Target evidence token 空间。
    cacheclip_scores = map_small_scores_to_large_tokens(
        small_scores=small_raw_scores,
        small_offsets=target_small["offsets"],
        small_evidence_positions=target_small["evidence_positions"],
        large_offsets=target_large["offsets"],
        large_evidence_positions=target_large["evidence_positions"],
    )

    # --------------------------------------------------------
    # 2. 大模型：Source / Target Full Prefill
    # --------------------------------------------------------
    source_full, source_latency = measure_time(
        lambda: full_prefill(
            model=large_model,
            input_ids_cpu=source_large["input_ids"],
            device=large_device,
            output_attentions=False,
        ),
        large_device,
    )

    target_full, target_latency = measure_time(
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
    # 3. Full Recompute
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

    rows = []

    rows.append(
        make_result_row(
            record=record,
            method="Full Recompute",
            method_type="full_recompute",
            is_oracle=False,
            repair_ratio=1.0,
            selected_token_count=len(target_evidence_positions),
            full_result=target_full,
            method_result=target_full,
            generated_answer=full_answer,
            full_answer=full_answer,
            rankings=full_rankings,
            top_k=args.top_k,
            latency=target_latency + full_decode_latency + full_rank_latency,
        )
    )

    # --------------------------------------------------------
    # 4. Full Long-Path KV Reuse：0% repair
    # --------------------------------------------------------
    direct_cache, direct_patch_latency = measure_time(
        lambda: build_hybrid_path_cache(
            source_cache=source_full["cache"],
            target_cache=target_full["cache"],
            source_full_path_span=source_full_path_span,
            target_full_path_span=target_full_path_span,
            target_evidence_positions=target_evidence_positions,
            selected_local_indices=[],
        ),
        large_device,
    )

    direct_result, direct_suffix_latency = measure_time(
        lambda: recompute_target_suffix(
            model=large_model,
            hybrid_cache=direct_cache,
            target_input_ids_cpu=target_large["input_ids"],
            target_full_path_span=target_full_path_span,
            device=large_device,
        ),
        large_device,
    )

    direct_answer, direct_decode_latency = measure_time(
        lambda: greedy_generate(
            model=large_model,
            tokenizer=large_tokenizer,
            initial_logits=direct_result["logits"],
            initial_cache=direct_result["cache"],
            initial_seq_len=direct_result["seq_len"],
            device=large_device,
            max_new_tokens=args.max_new_tokens,
        ),
        large_device,
    )

    direct_rankings, direct_rank_latency = measure_time(
        lambda: rank_candidates(
            model=large_model,
            tokenizer=large_tokenizer,
            initial_logits=direct_result["logits"],
            initial_cache=direct_result["cache"],
            initial_seq_len=direct_result["seq_len"],
            candidates=candidates,
            device=large_device,
        ),
        large_device,
    )

    rows.append(
        make_result_row(
            record=record,
            method="Full Long-Path KV Reuse",
            method_type="full_reuse",
            is_oracle=False,
            repair_ratio=0.0,
            selected_token_count=0,
            full_result=target_full,
            method_result=direct_result,
            generated_answer=direct_answer,
            full_answer=full_answer,
            rankings=direct_rankings,
            top_k=args.top_k,
            latency=(
                source_latency
                + target_latency
                + small_attention_latency
                + direct_patch_latency
                + direct_suffix_latency
                + direct_decode_latency
                + direct_rank_latency
            ),
        )
    )

    # --------------------------------------------------------
    # 5. Token-level CacheClip：10% / 20% / 30% / 40%
    # --------------------------------------------------------
    token_rows = []

    for repair_ratio in args.repair_ratios:
        selected_indices, budget = select_top_ratio_tokens(
            cacheclip_scores,
            repair_ratio,
        )

        hybrid_cache, patch_latency = measure_time(
            lambda: build_hybrid_path_cache(
                source_cache=source_full["cache"],
                target_cache=target_full["cache"],
                source_full_path_span=source_full_path_span,
                target_full_path_span=target_full_path_span,
                target_evidence_positions=target_evidence_positions,
                selected_local_indices=selected_indices,
            ),
            large_device,
        )

        cacheclip_result, suffix_latency = measure_time(
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
                initial_logits=cacheclip_result["logits"],
                initial_cache=cacheclip_result["cache"],
                initial_seq_len=cacheclip_result["seq_len"],
                device=large_device,
                max_new_tokens=args.max_new_tokens,
            ),
            large_device,
        )

        rankings, rank_latency = measure_time(
            lambda: rank_candidates(
                model=large_model,
                tokenizer=large_tokenizer,
                initial_logits=cacheclip_result["logits"],
                initial_cache=cacheclip_result["cache"],
                initial_seq_len=cacheclip_result["seq_len"],
                candidates=candidates,
                device=large_device,
            ),
            large_device,
        )

        rows.append(
            make_result_row(
                record=record,
                method=f"CacheClip Token Repair {repair_ratio:.0%}",
                method_type="cacheclip_token_oracle_patch",
                is_oracle=True,
                repair_ratio=repair_ratio,
                selected_token_count=budget,
                full_result=target_full,
                method_result=cacheclip_result,
                generated_answer=generated_answer,
                full_answer=full_answer,
                rankings=rankings,
                top_k=args.top_k,
                latency=(
                    source_latency
                    + target_latency
                    + small_attention_latency
                    + patch_latency
                    + suffix_latency
                    + decode_latency
                    + rank_latency
                ),
            )
        )

        selected_set = set(selected_indices)

        target_token_ids = target_large["input_ids"].tolist()

        for local_index, global_position in enumerate(
            target_evidence_positions
        ):
            token_id = int(target_token_ids[global_position])

            token_text = large_tokenizer.convert_ids_to_tokens(
                token_id
            )

            if token_text is None:
                token_text = str(token_id)

            token_rows.append(
                {
                    "sample_id": sample_id,
                    "repair_ratio": repair_ratio,
                    "path_local_index": local_index,
                    "global_token_position": global_position,
                    "token_id": token_id,
                    "token_label": token_text.replace("\n", "\\n"),
                    "hop_index": target_large["global_to_hop"][
                        global_position
                    ],
                    "cacheclip_score": float(
                        cacheclip_scores[local_index]
                    ),
                    "selected": int(
                        local_index in selected_set
                    ),
                }
            )

        del hybrid_cache
        del cacheclip_result

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metadata = {
        "sample_id": sample_id,
        "small_attention_layers": small_attention_result[
            "selected_layers"
        ],
        "evidence_token_count": len(target_evidence_positions),
        "source_path_start": source_path_start,
        "target_path_start": target_path_start,
        "path_position_shift": target_path_start - source_path_start,
    }

    del small_result
    del source_full
    del target_full

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return rows, pd.DataFrame(token_rows), metadata


# ============================================================
# 聚合结果
# ============================================================

def aggregate_results(result_df):
    grouped = (
        result_df.groupby(
            [
                "method",
                "method_type",
                "is_oracle",
                "repair_ratio",
            ],
            as_index=False,
        )
        .agg(
            n_samples=("sample_id", "nunique"),
            candidate_accuracy=("candidate_accuracy", "mean"),
            contains_gold_answer=("contains_gold_answer", "mean"),
            extracted_entity_em=("extracted_entity_em", "mean"),
            generation_match_full=("generation_match_full", "mean"),
            mean_gold_candidate_rank=("gold_candidate_rank", "mean"),
            mean_kl=("mean_kl", "mean"),
            mean_logits_cosine=("mean_logits_cosine", "mean"),
            mean_top1_match=("mean_top1_match", "mean"),
            mean_topk_overlap=("mean_topk_overlap", "mean"),
            mean_selected_token_count=("selected_token_count", "mean"),
            mean_analysis_latency_sec=("analysis_latency_sec", "mean"),
        )
    )

    def order(row):
        if row["method_type"] == "full_reuse":
            return 0.0

        if row["method_type"] == "cacheclip_token_oracle_patch":
            return 1.0 + row["repair_ratio"]

        if row["method_type"] == "full_recompute":
            return 3.0

        return 99.0

    grouped["method_order"] = grouped.apply(order, axis=1)

    return grouped.sort_values("method_order").reset_index(drop=True)


def aggregate_hop_selection(token_df):
    if len(token_df) == 0:
        return pd.DataFrame()

    return (
        token_df.groupby(
            [
                "repair_ratio",
                "hop_index",
            ],
            as_index=False,
        )
        .agg(
            n_tokens=("path_local_index", "count"),
            selected_token_rate=("selected", "mean"),
            mean_cacheclip_score=("cacheclip_score", "mean"),
        )
    )


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
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

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
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
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
            "默认 1：与原始 CacheClip 的最后一层 attention "
            "selector 更接近。设置为 4 可使用小模型最后四层平均。"
        ),
    )

    parser.add_argument(
        "--repair_ratios",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4],
        help=(
            "Token-level CacheClip repair ratios. "
            "Example: --repair_ratios 0.1 0.2 0.3 0.4"
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
        if not 0 < ratio < 1:
            raise ValueError(
                "repair_ratios must be strictly between 0 and 1. "
                "Full reuse and full recompute are already included."
            )

    benchmark_path = Path(args.benchmark_jsonl)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not benchmark_path.exists():
        raise FileNotFoundError(
            f"Benchmark not found: {benchmark_path}"
        )

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    dtype = parse_dtype(args.dtype)

    print("=" * 110)
    print("Token-Level CacheClip KV Repair Benchmark")
    print("=" * 110)
    print(f"Benchmark: {benchmark_path}")
    print(f"Output: {output_dir}")
    print(f"Primary model: {args.primary_model}")
    print(f"Auxiliary model: {args.auxiliary_model}")
    print(f"Small attention last K: {args.small_attention_last_k}")
    print(f"Repair ratios: {args.repair_ratios}")
    print("=" * 110)

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

    all_rows = []
    all_token_dfs = []
    all_metadata = []

    success_count = 0

    for index, record in enumerate(records, start=1):
        sample_id = record.get(
            "sample_id",
            f"sample_{index:05d}",
        )

        print(f"\n[{index}/{len(records)}] {sample_id}")

        try:
            rows, token_df, metadata = run_one_sample(
                record=record,
                small_tokenizer=small_tokenizer,
                small_model=small_model,
                small_device=small_device,
                large_tokenizer=large_tokenizer,
                large_model=large_model,
                large_device=large_device,
                args=args,
            )

            all_rows.extend(rows)
            all_token_dfs.append(token_df)
            all_metadata.append(metadata)

            success_count += 1

            print(
                f"  attention_layers={metadata['small_attention_layers']}, "
                f"evidence_tokens={metadata['evidence_token_count']}, "
                f"position_shift={metadata['path_position_shift']}"
            )

        except Exception as exc:
            print(f"[Warning] Failed sample {sample_id}: {exc}")

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(all_rows) == 0:
        raise RuntimeError("No sample completed successfully.")

    result_df = pd.DataFrame(all_rows)

    token_df = pd.concat(
        all_token_dfs,
        ignore_index=True,
    )

    aggregate_df = aggregate_results(result_df)
    hop_df = aggregate_hop_selection(token_df)

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

    hop_df.to_csv(
        output_dir / "hop_selection.csv",
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

    print("\n" + "=" * 110)
    print("Aggregate Results")
    print("=" * 110)
    print(
        aggregate_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\n" + "=" * 110)
    print("Hop Selection Summary")
    print("=" * 110)
    print(
        hop_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\n" + "=" * 110)
    print(
        f"Completed: {success_count}/{len(records)} samples"
    )
    print(f"Saved to: {output_dir.resolve()}")
    print("=" * 110)


if __name__ == "__main__":
    main()
