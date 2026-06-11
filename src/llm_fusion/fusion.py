"""Fuse token probability distributions from Ouro-1.4B and HRM-Text-1B."""

from __future__ import annotations

import math
from tokenizers import Tokenizer

from llm_fusion.token_matcher import TokenMatcher, Match


def softmax_top_k(logits: list[float], k: int) -> tuple[list[int], list[float]]:
    if not logits:
        return [], []
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])[:k]
    top_ids = [i for i, _ in indexed]
    top_vals = [v for _, v in indexed]
    max_val = max(top_vals)
    exps = [math.exp(v - max_val) for v in top_vals]
    total = sum(exps)
    probs = [e / total for e in exps]
    return top_ids, probs


class Fuser:
    def __init__(
        self,
        matcher: TokenMatcher,
        ouro_tok: Tokenizer,
        hrm_tok: Tokenizer,
        ouro_weight: float = 0.5,
        top_k: int = 50,
        threshold: float = 0.01,
    ):
        self.matcher = matcher
        self.ouro_tok = ouro_tok
        self.hrm_tok = hrm_tok
        self.ouro_weight = ouro_weight
        self.hrm_weight = 1.0 - ouro_weight
        self.top_k = top_k
        self.threshold = threshold

    def fuse_logits(
        self, ouro_logits: list[float], hrm_logits: list[float],
    ) -> list[tuple[int, float, str]]:
        ouro_top_ids, ouro_probs = softmax_top_k(ouro_logits, self.top_k)
        hrm_top_ids, hrm_probs = softmax_top_k(hrm_logits, self.top_k)

        fused: dict[int, float] = {}

        for tid, prob in zip(hrm_top_ids, hrm_probs):
            fused[tid] = fused.get(tid, 0.0) + prob * self.hrm_weight

        for oid, prob in zip(ouro_top_ids, ouro_probs):
            match = self.matcher.ouro_to_hrm(oid)
            if not match.target_ids:
                continue
            share = prob / len(match.target_ids)
            for tid in match.target_ids:
                fused[tid] = fused.get(tid, 0.0) + share * self.ouro_weight

        filtered = [(tid, p) for tid, p in fused.items() if p >= self.threshold]
        filtered.sort(key=lambda x: -x[1])

        return [(tid, p, self.hrm_tok.decode([tid])) for tid, p in filtered]

    def sample_token(
        self, ouro_logits: list[float], hrm_logits: list[float], temperature: float = 1.0,
    ) -> tuple[int, str, float]:
        import random
        candidates = self.fuse_logits(ouro_logits, hrm_logits)
        if not candidates:
            return 0, "", 0.0
        if temperature <= 0 or len(candidates) == 1:
            return candidates[0][0], candidates[0][2], candidates[0][1]
        probs = [p for _, p, _ in candidates]
        temp_probs = [math.log(max(p, 1e-10)) / temperature for p in probs]
        max_log = max(temp_probs)
        weights = [math.exp(lp - max_log) for lp in temp_probs]
        total = sum(weights)
        normalized = [w / total for w in weights]
        r = random.random()
        cumulative = 0.0
        for i, w in enumerate(normalized):
            cumulative += w
            if r <= cumulative:
                return candidates[i][0], candidates[i][2], candidates[i][1]
        return candidates[-1][0], candidates[-1][2], candidates[-1][1]
