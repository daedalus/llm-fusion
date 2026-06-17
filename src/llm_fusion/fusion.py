"""Fuse token probability distributions from Ouro-1.4B and HRM-Text-1B."""

from __future__ import annotations

import logging
import math
import random
from typing import TYPE_CHECKING

log = logging.getLogger(__name__)

if TYPE_CHECKING:
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


def compute_kl_torch(
    p_ids: list[int], p_probs: list[float],
    q_ids: list[int], q_probs: list[float],
) -> float:
    import torch

    q_map = dict(zip(q_ids, q_probs))
    p_t = torch.tensor(p_probs, dtype=torch.float32)
    q_t = torch.tensor([max(q_map.get(pid, 0.0), 1e-10) for pid in p_ids], dtype=torch.float32)
    mask = p_t > 0
    if not mask.any():
        return 0.0
    return float((p_t[mask] * (p_t[mask].log() - q_t[mask])).sum())


def softmax_top_k(logits: list[float], k: int) -> tuple[list[int], list[float]]:
    if not logits:
        return [], []
    import torch

    t = torch.tensor(logits, dtype=torch.float32)
    k = min(k, len(logits))
    topk_vals, topk_ids = torch.topk(t, k)
    probs = torch.softmax(topk_vals, dim=0)
    return topk_ids.tolist(), probs.tolist()


def softmax_top_k_torch(logits_t: "torch.Tensor", k: int) -> tuple[list[int], list[float]]:
    if logits_t.numel() == 0:
        return [], []
    k = min(k, logits_t.numel())
    topk_vals, topk_ids = torch.topk(logits_t.float(), k)
    probs = torch.softmax(topk_vals, dim=0)
    return topk_ids.tolist(), probs.tolist()


class Fuser:
    def __init__(
        self,
        matcher: TokenMatcher,
        ouro_tok: Tokenizer,
        hrm_tok: Tokenizer,
        ouro_weight: float = 0.5,
        top_k: int = 50,
        threshold: float = 0.01,
        strategy: str = "dynamic",
        cascade_threshold: float = 0.5,
        dynamic_initial_weight: float = 0.8,
        dynamic_final_weight: float = 0.2,
        dynamic_total_steps: int = 100,
    ) -> None:
        self.matcher = matcher
        self.ouro_tok = ouro_tok
        self.hrm_tok = hrm_tok
        self.ouro_weight = ouro_weight
        self.hrm_weight = 1.0 - ouro_weight
        self.top_k = top_k
        self.threshold = threshold
        valid = ("average", "product", "min-entropy", "cascade", "dynamic", "adaptive", "confidence", "hybrid")
        if strategy not in valid:
            raise ValueError(f"Unknown strategy: {strategy!r}")
        self.strategy = strategy
        self.cascade_threshold = cascade_threshold
        self.current_step = 0
        self.dynamic_initial_weight = dynamic_initial_weight
        self.dynamic_final_weight = dynamic_final_weight
        self.dynamic_total_steps = dynamic_total_steps

    def _fuse_logits_average(
        self,
        ouro_logits: list[float],
        hrm_logits: list[float],
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
        self,
        ouro_logits: list[float],
        hrm_logits: list[float],
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
        self,
        ouro_logits: list[float],
        hrm_logits: list[float],
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
        self,
        ouro_logits: list[float],
        hrm_logits: list[float],
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
        self,
        ouro_logits: list[float],
        hrm_logits: list[float],
    ) -> list[tuple[int, float, str]]:
        t = self.dynamic_total_steps
        s = min(self.current_step, t)
        ow = self.dynamic_initial_weight - (
            self.dynamic_initial_weight - self.dynamic_final_weight
        ) * s / max(t, 1)
        ow = max(self.dynamic_final_weight, min(self.dynamic_initial_weight, ow))
        hw = 1.0 - ow
        return self._fuse_logits_average(ouro_logits, hrm_logits, ow, hw)

    def _fuse_logits_adaptive(
        self,
        ouro_logits: list[float],
        hrm_logits: list[float],
    ) -> list[tuple[int, float, str]]:
        ouro_entropy = self._distribution_entropy(ouro_logits, self.top_k)
        hrm_entropy = self._distribution_entropy(hrm_logits, self.top_k)
        total = ouro_entropy + hrm_entropy
        if total < 1e-10:
            ow, hw = 0.5, 0.5
        else:
            ow = hrm_entropy / total
            hw = ouro_entropy / total
        return self._fuse_logits_average(ouro_logits, hrm_logits, ow, hw)

    def _fuse_logits_confidence(
        self,
        ouro_logits: list[float],
        hrm_logits: list[float],
    ) -> list[tuple[int, float, str]]:
        ouro_top_ids, ouro_probs = softmax_top_k(ouro_logits, 1)
        hrm_top_ids, hrm_probs = softmax_top_k(hrm_logits, 1)
        ouro_conf = ouro_probs[0] if ouro_probs else 0.0
        hrm_conf = hrm_probs[0] if hrm_probs else 0.0
        total = ouro_conf + hrm_conf
        if total < 1e-10:
            ow, hw = 0.5, 0.5
        else:
            ow = ouro_conf / total
            hw = hrm_conf / total
        return self._fuse_logits_average(ouro_logits, hrm_logits, ow, hw)

    def _fuse_logits_hybrid(
        self,
        ouro_logits: list[float],
        hrm_logits: list[float],
    ) -> list[tuple[int, float, str]]:
        t = self.dynamic_total_steps
        s = min(self.current_step, t)
        base_ow = self.dynamic_initial_weight - (
            self.dynamic_initial_weight - self.dynamic_final_weight
        ) * s / max(t, 1)
        base_ow = max(self.dynamic_final_weight, min(self.dynamic_initial_weight, base_ow))
        ouro_top_ids, ouro_probs = softmax_top_k(ouro_logits, 1)
        hrm_top_ids, hrm_probs = softmax_top_k(hrm_logits, 1)
        ouro_conf = ouro_probs[0] if ouro_probs else 0.0
        hrm_conf = hrm_probs[0] if hrm_probs else 0.0
        conf_total = ouro_conf + hrm_conf
        if conf_total < 1e-10:
            conf_ow = 0.5
        else:
            conf_ow = ouro_conf / conf_total
        ow = 0.7 * base_ow + 0.3 * conf_ow
        hw = 1.0 - ow
        return self._fuse_logits_average(ouro_logits, hrm_logits, ow, hw)

    def fuse_logits(
        self,
        ouro_logits: list[float],
        hrm_logits: list[float],
    ) -> list[tuple[int, float, str]]:
        if self.strategy == "product":
            return self._fuse_logits_product(ouro_logits, hrm_logits)
        if self.strategy == "min-entropy":
            return self._fuse_logits_minentropy(ouro_logits, hrm_logits)
        if self.strategy == "cascade":
            return self._fuse_logits_cascade(ouro_logits, hrm_logits)
        if self.strategy == "dynamic":
            return self._fuse_logits_dynamic(ouro_logits, hrm_logits)
        if self.strategy == "adaptive":
            return self._fuse_logits_adaptive(ouro_logits, hrm_logits)
        if self.strategy == "confidence":
            return self._fuse_logits_confidence(ouro_logits, hrm_logits)
        if self.strategy == "hybrid":
            return self._fuse_logits_hybrid(ouro_logits, hrm_logits)
        return self._fuse_logits_average(ouro_logits, hrm_logits)

    def model_distributions(
        self,
        ouro_logits: list[float],
        hrm_logits: list[float],
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
        self,
        ouro_logits: list[float],
        hrm_logits: list[float],
        temperature: float = 1.0,
        rng: random.Random | None = None,
    ) -> tuple[int, str, float]:
        if rng is None:
            rng = random.Random()
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
        r = rng.random()
        cumulative = 0.0
        for i, w in enumerate(normalized):
            cumulative += w
            if r <= cumulative:
                return candidates[i][0], candidates[i][2], candidates[i][1]
        return candidates[-1][0], candidates[-1][2], candidates[-1][1]

    def sample_token_pair(
        self,
        ouro_logits: list[float],
        hrm_logits: list[float],
        temperature: float = 1.0,
        rng: random.Random | None = None,
    ) -> tuple[int, int, str, float]:
        hrm_id, token_str, prob = self.sample_token(ouro_logits, hrm_logits, temperature, rng)
        match = self.matcher.map_sequence([hrm_id], "hrm")
        if match.target_ids:
            ouro_id = match.target_ids[0]
        else:
            ouro_id = 0
            log.warning("sample_token_pair: hrm_to_ouro returned empty for hrm[%d] %r", hrm_id, token_str)
        return hrm_id, ouro_id, token_str, prob
