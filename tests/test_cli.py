"""Tests for llm_fusion.cli."""

from __future__ import annotations

from llm_fusion.cli import build_parser


class TestBuildParser:
    def test_parser_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["hello world"])
        assert args.prompt == "hello world"
        assert args.model == "fused"
        assert args.temperature == 1.0
        assert args.top_k == 30
        assert args.ouro_weight == 0.5
        assert args.repetition_penalty == 1.0
        assert args.condition == "direct"
        assert args.max_new_tokens == 100
        assert args.local is False

    def test_parser_no_prompt(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.prompt is None

    def test_parser_custom_values(self):
        parser = build_parser()
        args = parser.parse_args([
            "--model", "ouro",
            "--temp", "0.5",
            "--top-k", "10",
            "--ouro-weight", "0.3",
            "--rep-penalty", "1.2",
            "--condition", "cot",
            "-n", "50",
            "--local",
            "test prompt",
        ])
        assert args.model == "ouro"
        assert args.temperature == 0.5
        assert args.top_k == 10
        assert args.ouro_weight == 0.3
        assert args.repetition_penalty == 1.2
        assert args.condition == "cot"
        assert args.max_new_tokens == 50
        assert args.local is True
        assert args.prompt == "test prompt"

    def test_parser_all_models(self):
        parser = build_parser()
        for m in ["fused", "ouro", "hrm"]:
            args = parser.parse_args(["--model", m, "p"])
            assert args.model == m

    def test_parser_all_conditions(self):
        parser = build_parser()
        for c in ["direct", "cot", "noisy", "synth"]:
            args = parser.parse_args(["--condition", c, "p"])
            assert args.condition == c

    def test_parser_short_model_flag(self):
        parser = build_parser()
        args = parser.parse_args(["-m", "ouro", "p"])
        assert args.model == "ouro"
