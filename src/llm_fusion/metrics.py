"""Fusion quality metrics — is the fused model better than its parents?"""

from __future__ import annotations

import math
from typing import Any


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
    ouro_model: Any,
    hrm_model: Any,
    ouro_tok: Any,
    hrm_tok: Any,
    fuser: Any,
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

        if hrm_prob > ouro_prob:
            oracle_wins += 1
        elif ouro_prob > hrm_prob:
            oracle_wins += 1
        else:
            oracle_wins += 0.5

        if fused_prob > max(ouro_prob, hrm_prob):
            fusion_wins += 1

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
    }
