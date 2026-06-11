"""Tests for llm_fusion.generate."""

from __future__ import annotations

from llm_fusion.generate import (
    format_hrm_prompt,
    strip_hrm_output,
    apply_repetition_penalty,
    OURO_EOS_ID,
    HRM_EOS_ID,
    patch_ouro_model,
)


class TestConstants:
    def test_ouro_eos_id(self):
        assert OURO_EOS_ID == 0

    def test_hrm_eos_id(self):
        assert HRM_EOS_ID == 11


class TestFormatPrompt:
    def test_format_hrm_prompt_direct(self):
        result = format_hrm_prompt("hello", "direct")
        assert result == "<|im_start|><|direct|>hello<|im_end|>"

    def test_format_hrm_prompt_cot(self):
        result = format_hrm_prompt("explain", "cot")
        assert "<|cot|>" in result

    def test_format_hrm_prompt_empty_text(self):
        result = format_hrm_prompt("", "direct")
        assert result == "<|im_start|><|direct|><|im_end|>"


class TestStripOutput:
    def test_strip_hrm_basic(self):
        assert strip_hrm_output("hello<|box_end|>") == "hello"

    def test_strip_hrm_with_im_tags(self):
        result = strip_hrm_output("<|im_start|>inner<|im_end|> outside")
        assert result == "outside"

    def test_strip_hrm_empty(self):
        assert strip_hrm_output("") == ""


class TestRepetitionPenalty:
    def test_no_penalty_at_one(self):
        logits = [1.0, 2.0, 3.0]
        assert apply_repetition_penalty(logits, {0}, 1.0) == logits

    def test_penalty_applied(self):
        logits = [1.0, 2.0, 3.0]
        result = apply_repetition_penalty(logits, {0}, 2.0)
        assert result[0] < logits[0]
        assert result[1:] == logits[1:]

    def test_penalty_negative_logit(self):
        logits = [-1.0, 2.0]
        result = apply_repetition_penalty(logits, {0}, 2.0)
        assert result[0] < logits[0]

    def test_penalty_empty_seen_ids(self):
        logits = [1.0, 2.0]
        assert apply_repetition_penalty(logits, set(), 2.0) == logits

    def test_penalty_id_out_of_range(self):
        logits = [1.0]
        result = apply_repetition_penalty(logits, {999}, 2.0)
        assert result == logits


class TestPatchOuroModel:
    def test_patch_sets_eager(self):
        class FakeConfig:
            _attn_implementation = None
        config = FakeConfig()
        patch_ouro_model(config)
        assert config._attn_implementation == "eager"
