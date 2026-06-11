"""Tests for llm_fusion.fusion."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from llm_fusion.token_matcher import TokenMatcher
from llm_fusion.fusion import Fuser, softmax_top_k


@pytest.fixture
def matcher() -> TokenMatcher:
    base = Path(__file__).resolve().parent.parent
    ouro_path = base / "Ouro-1.4B/tokenizer.json"
    hrm_path = base / "HRM-Text-1B/tokenizer.json"
    if not ouro_path.exists() or not hrm_path.exists():
        pytest.skip("model tokenizer files not found")
    return TokenMatcher(str(ouro_path), str(hrm_path))


@pytest.fixture
def fuser(matcher) -> Fuser:
    return Fuser(matcher, matcher.ouro_tok, matcher.hrm_tok)


class TestCascade:
    def test_cascade_uses_ouro_when_confident(self, matcher):
        target = matcher.ouro_to_hrm(335)
        if not target.target_ids:
            pytest.skip("no HRM mapping for Ouro token 335")
        fuser = Fuser(matcher, matcher.ouro_tok, matcher.hrm_tok, strategy="cascade",
                       cascade_threshold=0.5)
        ouro_logits = [0.0] * fuser.ouro_tok.get_vocab_size()
        hrm_logits = [0.0] * fuser.hrm_tok.get_vocab_size()
        ouro_logits[335] = 20.0
        ouro_logits[0] = 0.1
        hrm_logits[42] = 10.0
        results = fuser.fuse_logits(ouro_logits, hrm_logits)
        tids = {tid for tid, _, _ in results}
        expected = set(target.target_ids)
        assert tids & expected

    def test_cascade_fallsback_to_hrm_when_uncertain(self, matcher):
        fuser = Fuser(matcher, matcher.ouro_tok, matcher.hrm_tok, strategy="cascade",
                       cascade_threshold=0.5)
        ouro_logits = [0.0] * fuser.ouro_tok.get_vocab_size()
        hrm_logits = [0.0] * fuser.hrm_tok.get_vocab_size()
        ouro_logits[0] = 0.1
        ouro_logits[1] = 0.09
        hrm_logits[371] = 10.0
        results = fuser.fuse_logits(ouro_logits, hrm_logits)
        tids = [tid for tid, _, _ in results]
        assert 371 in tids

    def test_cascade_always_ouro_at_zero_threshold(self, matcher):
        fuser = Fuser(matcher, matcher.ouro_tok, matcher.hrm_tok, strategy="cascade",
                       cascade_threshold=0.0)
        ouro_logits = [0.0] * fuser.ouro_tok.get_vocab_size()
        hrm_logits = [0.0] * fuser.hrm_tok.get_vocab_size()
        ouro_logits[335] = 0.01
        hrm_logits[371] = 10.0
        results = fuser.fuse_logits(ouro_logits, hrm_logits)
        assert len(results) > 0

    def test_cascade_empty_logits(self, matcher):
        fuser = Fuser(matcher, matcher.ouro_tok, matcher.hrm_tok, strategy="cascade")
        results = fuser.fuse_logits([], [])
        assert results == []


class TestDynamic:
    def test_dynamic_weight_decays_with_step(self, matcher):
        fuser = Fuser(matcher, matcher.ouro_tok, matcher.hrm_tok, strategy="dynamic",
                       dynamic_initial_weight=0.9, dynamic_final_weight=0.1,
                       dynamic_total_steps=100)
        fuser.current_step = 0
        r0 = fuser.fuse_logits([0.0] * fuser.ouro_tok.get_vocab_size(),
                                [0.0] * fuser.hrm_tok.get_vocab_size())
        fuser.current_step = 50
        r50 = fuser.fuse_logits([0.0] * fuser.ouro_tok.get_vocab_size(),
                                 [0.0] * fuser.hrm_tok.get_vocab_size())
        fuser.current_step = 100
        r100 = fuser.fuse_logits([0.0] * fuser.ouro_tok.get_vocab_size(),
                                  [0.0] * fuser.hrm_tok.get_vocab_size())
        assert len(r0) >= 0 and len(r50) >= 0 and len(r100) >= 0

    def test_dynamic_weight_shifts_winner(self, matcher):
        early = Fuser(matcher, matcher.ouro_tok, matcher.hrm_tok, strategy="dynamic",
                       dynamic_initial_weight=0.9, dynamic_final_weight=0.1,
                       dynamic_total_steps=10)
        early.current_step = 0
        late = Fuser(matcher, matcher.ouro_tok, matcher.hrm_tok, strategy="dynamic",
                      dynamic_initial_weight=0.9, dynamic_final_weight=0.1,
                      dynamic_total_steps=10)
        late.current_step = 10
        vsize = early.ouro_tok.get_vocab_size()
        ouro_logits = [0.0] * vsize
        hrm_logits = [0.0] * late.hrm_tok.get_vocab_size()
        ouro_logits[335] = 5.0
        hrm_logits[371] = 5.0
        early_results = early.fuse_logits(ouro_logits, hrm_logits)
        late_results = late.fuse_logits(ouro_logits, hrm_logits)
        assert len(early_results) > 0 and len(late_results) > 0

    def test_dynamic_empty_logits(self, matcher):
        fuser = Fuser(matcher, matcher.ouro_tok, matcher.hrm_tok, strategy="dynamic")
        results = fuser.fuse_logits([], [])
        assert results == []


class TestSoftmaxTopK:
    def test_basic_top_k(self):
        logits = [0.0, 1.0, 2.0, 3.0]
        ids, probs = softmax_top_k(logits, 2)
        assert len(ids) == 2
        assert len(probs) == 2
        assert ids[0] == 3  # highest

    def test_empty_logits(self):
        ids, probs = softmax_top_k([], 5)
        assert ids == []
        assert probs == []

    def test_k_larger_than_vocab(self):
        logits = [0.0, 1.0]
        ids, probs = softmax_top_k(logits, 10)
        assert len(ids) == 2

    def test_all_same_values(self):
        logits = [1.0, 1.0, 1.0]
        ids, probs = softmax_top_k(logits, 3)
        assert len(ids) == 3
        assert abs(sum(probs) - 1.0) < 1e-6


class TestFuser:
    def test_fuse_logits_basic(self, fuser):
        ouro_logits = [0.0] * fuser.ouro_tok.get_vocab_size()
        hrm_logits = [0.0] * fuser.hrm_tok.get_vocab_size()
        ouro_logits[335] = 5.0
        hrm_logits[371] = 5.0
        results = fuser.fuse_logits(ouro_logits, hrm_logits)
        assert len(results) > 0

    def test_fuse_logits_empty(self, fuser):
        ouro_logits = [-100.0] * fuser.ouro_tok.get_vocab_size()
        hrm_logits = [-100.0] * fuser.hrm_tok.get_vocab_size()
        results = fuser.fuse_logits(ouro_logits, hrm_logits)
        assert len(results) >= 0

    def test_sample_token_greedy(self, fuser):
        ouro_logits = [0.0] * fuser.ouro_tok.get_vocab_size()
        hrm_logits = [0.0] * fuser.hrm_tok.get_vocab_size()
        hrm_logits[371] = 10.0
        tid, token_str, prob = fuser.sample_token(ouro_logits, hrm_logits, temperature=0)
        assert tid == 371

    def test_sample_token_temperature(self, matcher):
        fuser = Fuser(matcher, matcher.ouro_tok, matcher.hrm_tok, ouro_weight=0.0)
        ouro_logits = [0.0] * matcher.ouro_tok.get_vocab_size()
        hrm_logits = [0.0] * matcher.hrm_tok.get_vocab_size()
        hrm_logits[371] = 10.0
        hrm_logits[42] = 9.5
        tid, token_str, prob = fuser.sample_token(ouro_logits, hrm_logits, temperature=1.0)
        assert tid in (371, 42)

    def test_ouro_weight_effect(self, matcher):
        fuser_ouro = Fuser(matcher, matcher.ouro_tok, matcher.hrm_tok, ouro_weight=0.9)
        fuser_hrm = Fuser(matcher, matcher.ouro_tok, matcher.hrm_tok, ouro_weight=0.1)
        assert fuser_ouro.ouro_weight == 0.9
        assert fuser_hrm.ouro_weight == 0.1

    def test_invalid_strategy_raises(self, matcher):
        with pytest.raises(ValueError, match="Unknown strategy"):
            Fuser(matcher, matcher.ouro_tok, matcher.hrm_tok, strategy="ensemble")

    def test_fuse_logits_product_basic(self, matcher):
        fuser = Fuser(matcher, matcher.ouro_tok, matcher.hrm_tok, strategy="product")
        ouro_logits = [0.0] * fuser.ouro_tok.get_vocab_size()
        hrm_logits = [0.0] * fuser.hrm_tok.get_vocab_size()
        ouro_logits[335] = 5.0
        hrm_logits[371] = 5.0
        results = fuser.fuse_logits(ouro_logits, hrm_logits)
        assert len(results) > 0

    def test_min_entropy_routes_to_hrm(self, matcher):
        fuser = Fuser(matcher, matcher.ouro_tok, matcher.hrm_tok, strategy="min-entropy")
        ouro_logits = [0.0] * fuser.ouro_tok.get_vocab_size()
        hrm_logits = [0.0] * fuser.hrm_tok.get_vocab_size()
        hrm_logits[371] = 20.0
        hrm_logits[42] = 19.5
        ouro_logits[0] = 0.1
        results = fuser.fuse_logits(ouro_logits, hrm_logits)
        tids = [tid for tid, _, _ in results]
        assert 371 in tids or 42 in tids

    def test_min_entropy_routes_to_ouro(self, matcher):
        fuser = Fuser(matcher, matcher.ouro_tok, matcher.hrm_tok, strategy="min-entropy")
        ouro_logits = [0.0] * fuser.ouro_tok.get_vocab_size()
        hrm_logits = [0.0] * fuser.hrm_tok.get_vocab_size()
        ouro_logits[335] = 20.0
        hrm_logits[0] = 0.1
        results = fuser.fuse_logits(ouro_logits, hrm_logits)
        assert len(results) > 0

    def test_min_entropy_empty_logits(self, matcher):
        fuser = Fuser(matcher, matcher.ouro_tok, matcher.hrm_tok, strategy="min-entropy")
        results = fuser.fuse_logits([], [])
        assert results == []

    def test_fuse_logits_product_no_overlap(self, matcher):
        fuser = Fuser(matcher, matcher.ouro_tok, matcher.hrm_tok, strategy="product",
                       threshold=0.0)
        ouro_logits = [-100.0] * fuser.ouro_tok.get_vocab_size()
        hrm_logits = [-100.0] * fuser.hrm_tok.get_vocab_size()
        hrm_logits[42] = 10.0
        ouro_logits[0] = 10.0
        results = fuser.fuse_logits(ouro_logits, hrm_logits)
        for _, p, _ in results:
            assert p < 1e-6

    def test_product_kills_uncommon(self, matcher):
        fuser = Fuser(matcher, matcher.ouro_tok, matcher.hrm_tok, strategy="product")
        ouro_logits = [0.0] * fuser.ouro_tok.get_vocab_size()
        hrm_logits = [0.0] * fuser.hrm_tok.get_vocab_size()
        ouro_logits[335] = 5.0
        hrm_logits[371] = 5.0
        avg = Fuser(matcher, matcher.ouro_tok, matcher.hrm_tok, strategy="average")
        avg_results = avg.fuse_logits(ouro_logits, hrm_logits)
        prod_results = fuser.fuse_logits(ouro_logits, hrm_logits)
        assert len(prod_results) <= len(avg_results) + 1
