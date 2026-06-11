"""Tests for llm_fusion.metrics."""

from __future__ import annotations

import math

import pytest

from llm_fusion.metrics import compare_distributions, fusion_gain, parent_prob_for_token


class TestFusionGain:
    def test_positive_gain(self) -> None:
        g = fusion_gain(0.8, 0.5, 0.3)
        expected = math.log(0.8) - math.log(0.5)
        assert g == pytest.approx(expected)

    def test_negative_gain(self) -> None:
        g = fusion_gain(0.3, 0.5, 0.8)
        expected = math.log(0.3) - math.log(0.8)
        assert g == pytest.approx(expected)

    def test_zero_gain_when_equal(self) -> None:
        g = fusion_gain(0.5, 0.5, 0.3)
        assert g == 0.0

    def test_zero_fused_prob_returns_zero(self) -> None:
        g = fusion_gain(0.0, 0.5, 0.3)
        assert g == 0.0

    def test_best_parent_zero_returns_zero(self) -> None:
        g = fusion_gain(0.5, 0.0, 0.0)
        assert g == 0.0

    def test_fusion_worse_than_either(self) -> None:
        g = fusion_gain(0.1, 0.6, 0.7)
        expected = math.log(0.1) - math.log(0.7)
        assert g == pytest.approx(expected)


class TestParentProbForToken:
    def test_finds_prob_in_top_k(self) -> None:
        logits = [0.0, 1.0, 10.0, 5.0, 0.5]
        prob = parent_prob_for_token(logits, 2, k=5)
        assert prob > 0.5

    def test_not_in_top_k_returns_zero(self) -> None:
        logits = [100.0, 0.0, 0.0]
        prob = parent_prob_for_token(logits, 3, k=2)
        assert prob == 0.0

    def test_empty_logits(self) -> None:
        prob = parent_prob_for_token([], 0)
        assert prob == 0.0


class TestCompareDistributions:
    def test_basic_comparison(self) -> None:
        ouro = [5.0, 3.0, 1.0]
        hrm = [4.0, 4.0, 4.0]
        result = compare_distributions(ouro, hrm, ouro_top_k=3, hrm_top_k=3)
        assert "ouro_entropy" in result
        assert "hrm_entropy" in result
        assert "overlap_size" in result
        assert "kl_ouro_to_hrm" in result
        assert "kl_hrm_to_ouro" in result
        assert result["overlap_size"] >= 0

    def test_same_distributions(self) -> None:
        logits = [10.0, 5.0, 2.0]
        result = compare_distributions(logits, logits, ouro_top_k=3, hrm_top_k=3)
        assert result["overlap_size"] == 3
        assert result["kl_ouro_to_hrm"] == pytest.approx(0.0, abs=1e-6)

    def test_no_overlap(self) -> None:
        ouro = [10.0, 0.0, 0.0]
        hrm = [0.0, 0.0, 10.0]
        result = compare_distributions(ouro, hrm, ouro_top_k=1, hrm_top_k=1)
        assert result["overlap_size"] == 0
        assert result["kl_ouro_to_hrm"] > 0
        assert result["kl_hrm_to_ouro"] > 0
