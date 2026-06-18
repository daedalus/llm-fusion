"""Fusion quality metrics — is the fused model better than its parents?"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llm_fusion.fusion import Fuser
    from llm_fusion.loader import CausalLM


def fusion_gain(
    fused_prob: float,
    ouro_prob: float,
    hrm_prob: float,
) -> float:
    """How much fusion boosts over the best parent (positive = better).

    gain = log P_fused(token) - max(log P_ouro(token), log P_hrm(token))
    """
    best_parent = max(ouro_prob, hrm_prob)
    if fused_prob <= 0 or best_parent <= 0:
        return 0.0
    return math.log(fused_prob) - math.log(best_parent)


def parent_prob_for_token(
    logits: list[float],
    tid: int,
    k: int = 100,
) -> float:
    """Get the softmax probability of a specific token ID from logits."""
    from llm_fusion.fusion import softmax_top_k

    ids, probs = softmax_top_k(logits, k)
    for iid, p in zip(ids, probs):
        if iid == tid:
            return p
    return 0.0


def topk_accuracy(
    logits: list[float],
    target_tid: int,
    k: int,
) -> bool:
    """Is the target token in the top-k candidates?"""
    from llm_fusion.fusion import softmax_top_k

    ids, _ = softmax_top_k(logits, k)
    return target_tid in ids


def agreement_rate(
    ouro_logits: list[float],
    hrm_logits: list[float],
    k: int = 1,
) -> bool:
    """Do both models agree on the top-1 token?"""
    from llm_fusion.fusion import softmax_top_k

    ouro_ids, _ = softmax_top_k(ouro_logits, k)
    hrm_ids, _ = softmax_top_k(hrm_logits, k)
    if not ouro_ids or not hrm_ids:
        return False
    return ouro_ids[0] == hrm_ids[0]


def top1_accuracy(
    logits: list[float],
    target_tid: int,
) -> bool:
    """Does the model's greedy top-1 pick match the target?"""
    if not logits:
        return False
    return max(range(len(logits)), key=lambda i: logits[i]) == target_tid


def fusion_regret(
    fused_prob: float,
    ouro_prob: float,
    hrm_prob: float,
) -> float:
    """How much fusion hurts vs the better parent (positive = worse).

    regret = max(log P_ouro, log P_hrm) - log P_fused
    """
    best_parent = max(ouro_prob, hrm_prob)
    if best_parent <= 0 or fused_prob <= 0:
        return 0.0
    return math.log(best_parent) - math.log(fused_prob)


def contribution_ratio(
    fused_prob_per_token: dict[int, float],
    matched_ouro_prob_per_token: dict[int, float],
) -> float:
    """Fraction of fused probability mass that came from Ouro (0=all HRM, 1=all Ouro)."""
    total_fused = sum(fused_prob_per_token.values())
    if total_fused <= 0:
        return 0.5
    total_ouro = sum(matched_ouro_prob_per_token.get(tid, 0.0) for tid in fused_prob_per_token)
    return total_ouro / total_fused


def entropy_delta(
    fused_entropy: float,
    ouro_entropy: float,
    hrm_entropy: float,
) -> float:
    """Fused entropy minus average parent entropy. Negative = fusion concentrates."""
    return fused_entropy - 0.5 * (ouro_entropy + hrm_entropy)


def compute_entropy(probs: list[float]) -> float:
    """Shannon entropy of a probability distribution."""
    return -sum(p * math.log(max(p, 1e-10)) for p in probs if p > 0)


def calibration_error(
    predicted_probs: list[float],
    outcomes: list[bool],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error — gap between predicted confidence and observed accuracy."""
    if not predicted_probs:
        return 0.0
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for p, o in zip(predicted_probs, outcomes):
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, o))
    ece = 0.0
    total = len(predicted_probs)
    for bin_items in bins:
        if not bin_items:
            continue
        avg_p = sum(p for p, _ in bin_items) / len(bin_items)
        avg_o = sum(1.0 for _, o in bin_items if o) / len(bin_items)
        ece += len(bin_items) / total * abs(avg_p - avg_o)
    return ece


def token_diversity(token_ids: list[int]) -> float:
    """Fraction of unique tokens in a sequence (1.0 = all unique, low = repetitive)."""
    if not token_ids:
        return 0.0
    return len(set(token_ids)) / len(token_ids)


def compare_distributions(
    ouro_logits: list[float],
    hrm_logits: list[float],
    ouro_top_k: int = 50,
    hrm_top_k: int = 50,
) -> dict[str, Any]:
    """Compare Ouro and HRM distributions: entropy, overlap, agreement."""
    from llm_fusion.fusion import compute_kl, softmax_top_k

    ouro_ids, ouro_probs = softmax_top_k(ouro_logits, ouro_top_k)
    hrm_ids, hrm_probs = softmax_top_k(hrm_logits, hrm_top_k)

    ouro_set = set(ouro_ids)
    overlap = ouro_set & set(hrm_ids)

    ouro_ent = -sum(p * math.log(max(p, 1e-10)) for p in ouro_probs)
    hrm_ent = -sum(p * math.log(max(p, 1e-10)) for p in hrm_probs)

    ouro_dict = dict(zip(ouro_ids, ouro_probs))
    hrm_dict = dict(zip(hrm_ids, hrm_probs))

    return {
        "ouro_entropy": ouro_ent,
        "hrm_entropy": hrm_ent,
        "overlap_size": len(overlap),
        "kl_ouro_to_hrm": compute_kl(ouro_dict, hrm_dict),
        "kl_hrm_to_ouro": compute_kl(hrm_dict, ouro_dict),
    }


def evaluate_text(
    text: str,
    ouro_model: CausalLM,
    hrm_model: CausalLM,
    ouro_tok: Any,
    hrm_tok: Any,
    fuser: Fuser | None,
    device: str = "cpu",
    max_tokens: int = 100,
) -> dict[str, Any]:
    """Score a text under all three configurations: ouro, hrm, fused.

    Returns aggregate metrics showing whether fusion improves over parents.
    """
    import torch

    hrm_ids = hrm_tok.encode(text).ids

    total_gain = 0.0
    total_ouro_logprob = 0.0
    total_hrm_logprob = 0.0
    total_fused_logprob = 0.0
    oracle_wins = 0.0
    fusion_wins = 0.0
    n_tokens = 0.0

    topk_ouro = 0.0
    topk_hrm = 0.0
    topk_fused = 0.0
    agree_count = 0.0
    regret_sum = 0.0
    top1_ouro = 0.0
    top1_hrm = 0.0
    top1_fused = 0.0
    probs_for_cal = []
    outcomes_for_cal = []
    per_position_ppl = []

    seq = hrm_ids[:max_tokens] if len(hrm_ids) > max_tokens else hrm_ids

    for t in range(1, len(seq)):
        target_tid = seq[t]

        prefix = hrm_tok.decode(seq[:t])
        ouro_prefix_ids = ouro_tok.encode(prefix).ids or [0]

        with torch.no_grad():
            ouro_out = ouro_model(
                torch.tensor([ouro_prefix_ids], device=device),
            )
            hrm_out = hrm_model(
                torch.tensor([seq[:t]], device=device),
                token_type_ids=torch.ones(t, dtype=torch.long, device=device).unsqueeze(0),
            )

        ouro_logits = ouro_out.logits[0, -1, :].tolist()
        hrm_logits = hrm_out.logits[0, -1, :].tolist()

        candidates = fuser.fuse_logits(ouro_logits, hrm_logits)
        fused_prob = 0.0
        for tid, p, _ in candidates:
            if tid == target_tid:
                fused_prob = p
                break

        ouro_prob = parent_prob_for_token(ouro_logits, target_tid)
        hrm_prob = parent_prob_for_token(hrm_logits, target_tid)

        gain = fusion_gain(fused_prob, ouro_prob, hrm_prob)

        total_gain += gain
        total_ouro_logprob += math.log(max(ouro_prob, 1e-10))
        total_hrm_logprob += math.log(max(hrm_prob, 1e-10))
        total_fused_logprob += math.log(max(fused_prob, 1e-10))
        n_tokens += 1

        ouro_top1_tid = max(range(len(ouro_logits)), key=lambda i: ouro_logits[i])
        hrm_top1_tid = max(range(len(hrm_logits)), key=lambda i: hrm_logits[i])
        if ouro_prob >= hrm_prob:
            better_parent_top1 = ouro_top1_tid
        else:
            better_parent_top1 = hrm_top1_tid
        if target_tid == better_parent_top1:
            oracle_wins += 1

        if fused_prob > max(ouro_prob, hrm_prob):
            fusion_wins += 1

        topk_ouro += topk_accuracy(ouro_logits, target_tid, 10)
        topk_hrm += topk_accuracy(hrm_logits, target_tid, 10)
        topk_fused += fused_prob > 0
        agree_count += agreement_rate(ouro_logits, hrm_logits)
        top1_ouro += top1_accuracy(ouro_logits, target_tid)
        top1_hrm += top1_accuracy(hrm_logits, target_tid)
        top1_fused += fused_prob == max(p for _, p, _ in candidates) if candidates else False
        regret_sum += fusion_regret(fused_prob, ouro_prob, hrm_prob)

        probs_for_cal.append(fused_prob)
        outcomes_for_cal.append(fused_prob > 0)
        per_position_ppl.append(math.exp(-math.log(max(fused_prob, 1e-10))))

    avg_gain = total_gain / max(n_tokens, 1)
    ouro_ppl = math.exp(-total_ouro_logprob / max(n_tokens, 1))
    hrm_ppl = math.exp(-total_hrm_logprob / max(n_tokens, 1))
    fused_ppl = math.exp(-total_fused_logprob / max(n_tokens, 1))
    oracle_rate = oracle_wins / max(n_tokens, 1)

    return {
        "n_tokens": n_tokens,
        "avg_fusion_gain": avg_gain,
        "fusion_wins": fusion_wins,
        "fusion_win_rate": fusion_wins / max(n_tokens, 1),
        "oracle_rate": oracle_rate,
        "ouro_ppl": ouro_ppl,
        "hrm_ppl": hrm_ppl,
        "fused_ppl": fused_ppl,
        "ppl_improvement_vs_ouro": (ouro_ppl - fused_ppl) / ouro_ppl * 100,
        "ppl_improvement_vs_hrm": (hrm_ppl - fused_ppl) / hrm_ppl * 100,
        "top10_accuracy_ouro": topk_ouro / max(n_tokens, 1),
        "top10_accuracy_hrm": topk_hrm / max(n_tokens, 1),
        "top10_accuracy_fused": topk_fused / max(n_tokens, 1),
        "top1_accuracy_ouro": top1_ouro / max(n_tokens, 1),
        "top1_accuracy_hrm": top1_hrm / max(n_tokens, 1),
        "top1_accuracy_fused": top1_fused / max(n_tokens, 1),
        "agreement_rate": agree_count / max(n_tokens, 1),
        "avg_fusion_regret": regret_sum / max(n_tokens, 1),
        "calibration_error": calibration_error(probs_for_cal, outcomes_for_cal),
        "per_position_ppl": per_position_ppl,
    }
