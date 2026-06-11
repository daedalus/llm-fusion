"""Fuse token probability distributions from Ouro-1.4B and HRM-Text-1B."""

from __future__ import annotations

import math

from tokenizers import Tokenizer

from llm_fusion.token_matcher import TokenMatcher


def compute_kl(p: dict[int, float], q: dict[int, float]) -> float:
    all_ids = set(p) | set(q)
    kl = 0.0
    for tid in all_ids:
        p_prob = p.get(tid, 0.0)
        if p_prob == 0.0:
            continue
        q_prob = max(q.get(tid, 0.0), 1e-10)
        kl += p_prob * math.log(p_prob / q_prob)
    return kl


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
        strategy: str = "average",
        cascade_threshold: float = 0.5,
        dynamic_initial_weight: float = 0.8,
        dynamic_final_weight: float = 0.2,
        dynamic_total_steps: int = 100,
    ):
        self.matcher = matcher
        self.ouro_tok = ouro_tok
        self.hrm_tok = hrm_tok
        self.ouro_weight = ouro_weight
        self.hrm_weight = 1.0 - ouro_weight
        self.top_k = top_k
        self.threshold = threshold
        valid = ("average", "product", "min-entropy", "cascade", "dynamic")
        if strategy not in valid:
            raise ValueError(f"Unknown strategy: {strategy!r}")
        self.strategy = strategy
        self.cascade_threshold = cascade_threshold
        self.current_step = 0
        self.dynamic_initial_weight = dynamic_initial_weight
        self.dynamic_final_weight = dynamic_final_weight
        self.dynamic_total_steps = dynamic_total_steps

    def _fuse_logits_average(
        self, ouro_logits: list[float], hrm_logits: list[float],
        ouro_weight: float | None = None,
        hrm_weight: float | None = None,
    ) -> list[tuple[int, float, str]]:
        ow = self.ouro_weight if ouro_weight is None else ouro_weight
        hw = self.hrm_weight if hrm_weight is None else hrm_weight
        ouro_top_ids, ouro_probs = softmax_top_k(ouro_logits, self.top_k)
        hrm_top_ids, hrm_probs = softmax_top_k(hrm_logits, self.top_k)

        fused: dict[int, float] = {}

        for tid, prob in zip(hrm_top_ids, hrm_probs):
            fused[tid] = fused.get(tid, 0.0) + prob * hw

        for oid, prob in zip(ouro_top_ids, ouro_probs):
            match = self.matcher.ouro_to_hrm(oid)
            if not match.target_ids:
                continue
            share = prob / len(match.target_ids)
            for tid in match.target_ids:
                fused[tid] = fused.get(tid, 0.0) + share * ow

        filtered = [(tid, p) for tid, p in fused.items() if p >= self.threshold]
        filtered.sort(key=lambda x: -x[1])
        return [(tid, p, self.hrm_tok.decode([tid])) for tid, p in filtered]

    def _fuse_logits_product(
        self, ouro_logits: list[float], hrm_logits: list[float],
    ) -> list[tuple[int, float, str]]:
        ouro_top_ids, ouro_probs = softmax_top_k(ouro_logits, self.top_k)
        hrm_top_ids, hrm_probs = softmax_top_k(hrm_logits, self.top_k)

        ouro_given_hrm: dict[int, float] = {}
        for oid, prob in zip(ouro_top_ids, ouro_probs):
            match = self.matcher.ouro_to_hrm(oid)
            if not match.target_ids:
                continue
            share = prob / len(match.target_ids)
            for tid in match.target_ids:
                ouro_given_hrm[tid] = ouro_given_hrm.get(tid, 0.0) + share

        hrm_probs_dict = dict(zip(hrm_top_ids, hrm_probs))

        all_ids = set(ouro_given_hrm) | set(hrm_probs_dict)
        fused = {}
        for tid in all_ids:
            p_ouro = ouro_given_hrm.get(tid, 0.0)
            p_hrm = hrm_probs_dict.get(tid, 0.0)
            p = p_ouro * p_hrm
            if p >= self.threshold:
                fused[tid] = p

        filtered = sorted(fused.items(), key=lambda x: -x[1])
        return [(tid, p, self.hrm_tok.decode([tid])) for tid, p in filtered]

    @staticmethod
    def _distribution_entropy(logits: list[float], k: int) -> float:
        _, probs = softmax_top_k(logits, k)
        if not probs:
            return float("inf")
        return -sum(p * math.log(max(p, 1e-10)) for p in probs)

    def _fuse_logits_minentropy(
        self, ouro_logits: list[float], hrm_logits: list[float],
    ) -> list[tuple[int, float, str]]:
        ouro_entropy = self._distribution_entropy(ouro_logits, self.top_k)
        hrm_entropy = self._distribution_entropy(hrm_logits, self.top_k)

        if hrm_entropy < ouro_entropy:
            ids, probs = softmax_top_k(hrm_logits, self.top_k)
            filtered = [(tid, p) for tid, p in zip(ids, probs) if p >= self.threshold]
            return [(tid, p, self.hrm_tok.decode([tid])) for tid, p in filtered]

        ids, probs = softmax_top_k(ouro_logits, self.top_k)
        fused: dict[int, float] = {}
        for oid, prob in zip(ids, probs):
            match = self.matcher.ouro_to_hrm(oid)
            if not match.target_ids:
                continue
            share = prob / len(match.target_ids)
            for tid in match.target_ids:
                fused[tid] = fused.get(tid, 0.0) + share
        filtered = [(tid, p) for tid, p in fused.items() if p >= self.threshold]
        filtered.sort(key=lambda x: -x[1])
        return [(tid, p, self.hrm_tok.decode([tid])) for tid, p in filtered]

    def _fuse_logits_cascade(
        self, ouro_logits: list[float], hrm_logits: list[float],
    ) -> list[tuple[int, float, str]]:
        ids, probs = softmax_top_k(ouro_logits, self.top_k)
        if probs and probs[0] >= self.cascade_threshold:
            fused: dict[int, float] = {}
            for oid, prob in zip(ids, probs):
                match = self.matcher.ouro_to_hrm(oid)
                if not match.target_ids:
                    continue
                share = prob / len(match.target_ids)
                for tid in match.target_ids:
                    fused[tid] = fused.get(tid, 0.0) + share
            filtered = [(tid, p) for tid, p in fused.items() if p >= self.threshold]
            filtered.sort(key=lambda x: -x[1])
            return [(tid, p, self.hrm_tok.decode([tid])) for tid, p in filtered]

        ids, probs = softmax_top_k(hrm_logits, self.top_k)
        filtered = [(tid, p) for tid, p in zip(ids, probs) if p >= self.threshold]
        return [(tid, p, self.hrm_tok.decode([tid])) for tid, p in filtered]

    def _fuse_logits_dynamic(
        self, ouro_logits: list[float], hrm_logits: list[float],
    ) -> list[tuple[int, float, str]]:
        t = self.dynamic_total_steps
        s = min(self.current_step, t)
        ow = self.dynamic_initial_weight - (self.dynamic_initial_weight - self.dynamic_final_weight) * s / max(t, 1)
        ow = max(self.dynamic_final_weight, min(self.dynamic_initial_weight, ow))
        hw = 1.0 - ow
        return self._fuse_logits_average(ouro_logits, hrm_logits, ow, hw)

    def fuse_logits(
        self, ouro_logits: list[float], hrm_logits: list[float],
    ) -> list[tuple[int, float, str]]:
        if self.strategy == "product":
            return self._fuse_logits_product(ouro_logits, hrm_logits)
        if self.strategy == "min-entropy":
            return self._fuse_logits_minentropy(ouro_logits, hrm_logits)
        if self.strategy == "cascade":
            return self._fuse_logits_cascade(ouro_logits, hrm_logits)
        if self.strategy == "dynamic":
            return self._fuse_logits_dynamic(ouro_logits, hrm_logits)
        return self._fuse_logits_average(ouro_logits, hrm_logits)

    def model_distributions(
        self, ouro_logits: list[float], hrm_logits: list[float],
    ) -> tuple[dict[int, float], dict[int, float]]:
        ouro_top_ids, ouro_probs = softmax_top_k(ouro_logits, self.top_k)
        hrm_top_ids, hrm_probs = softmax_top_k(hrm_logits, self.top_k)
        ouro_mapped: dict[int, float] = {}
        for oid, prob in zip(ouro_top_ids, ouro_probs):
            match = self.matcher.ouro_to_hrm(oid)
            if not match.target_ids:
                continue
            share = prob / len(match.target_ids)
            for tid in match.target_ids:
                ouro_mapped[tid] = ouro_mapped.get(tid, 0.0) + share
        hrm_dict = dict(zip(hrm_top_ids, hrm_probs))
        return ouro_mapped, hrm_dict

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
