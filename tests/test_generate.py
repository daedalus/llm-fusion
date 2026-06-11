"""Tests for llm_fusion.generate."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from llm_fusion.generate import (
    compute_perplexity,
    compute_fused_perplexity,
    format_hrm_prompt,
    strip_hrm_output,
    apply_repetition_penalty,
    OURO_EOS_ID,
    HRM_EOS_ID,
    patch_ouro_model,
)


@dataclass
class FakeOutput:
    logits: torch.Tensor


def _make_tok(token_ids: list[int]):
    class FakeTok:
        def encode(self, text):
            class FakeEnc:
                ids = list(token_ids)
            return FakeEnc()
    return FakeTok()


class TestComputePerplexity:
    def test_uniform_model(self):
        model = lambda inp: FakeOutput(torch.zeros(1, inp.size(1), 1000))
        ppl = compute_perplexity("test", model, _make_tok([0, 1, 2, 3, 4]), device="cpu")
        assert 900 < ppl < 1100

    def test_empty_text(self):
        model = lambda inp: FakeOutput(torch.zeros(1, 1, 100))
        ppl = compute_perplexity("", model, _make_tok([]), device="cpu")
        assert ppl == float("inf")

    def test_perfect_model(self):
        def model(inp):
            logits = torch.full((1, inp.size(1), 1000), -100.0)
            for i in range(inp.size(1) - 1):
                logits[0, i, inp[0, i + 1].item()] = 100.0
            return FakeOutput(logits)
        ppl = compute_perplexity("test", model, _make_tok([0, 1, 2, 3]), device="cpu")
        assert ppl < 1.1


class FakeModel:
    def __init__(self, vocab_size=50000):
        self.vocab_size = vocab_size

    def __call__(self, input_ids, **kwargs):
        return FakeOutput(torch.zeros(1, input_ids.size(1), self.vocab_size))


def _make_simple_tok(vocab_size=50000):
    class Tok:
        def encode(self, text):
            class Enc:
                ids = [0, 1, 2, 42]
            return Enc()

        def decode(self, ids):
            return " ".join(str(i) for i in ids)
    return Tok()


class MockMatcher:
    class MockMatch:
        target_ids = [42]

    def ouro_to_hrm(self, oid):
        return self.MockMatch()


class TestComputeFusedPerplexity:
    def test_fused_uniform(self):
        from llm_fusion.fusion import Fuser
        ouro_model = FakeModel()
        hrm_model = FakeModel()
        ouro_tok = _make_simple_tok(1000)
        hrm_tok = _make_simple_tok(1000)
        matcher = MockMatcher()
        fuser = Fuser(matcher, ouro_tok, hrm_tok, strategy="average")
        ppl = compute_fused_perplexity("test", ouro_model, hrm_model, ouro_tok, hrm_tok,
                                        fuser, device="cpu")
        assert ppl > 1.0


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
