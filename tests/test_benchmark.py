"""Tests for llm_fusion.benchmark."""

from __future__ import annotations

from llm_fusion.benchmark import (
    ROBUSTNESS_BATTERY,
    BenchmarkResult,
    RobustnessResult,
    format_robustness_table,
    format_table,
    maybe_get_memory_mb,
)


class TestBenchmarkResult:
    def test_defaults(self) -> None:
        r = BenchmarkResult(model="ouro")
        assert r.model == "ouro"
        assert r.tokens_per_sec == 0.0

    def test_tokens_per_sec_computed(self) -> None:
        r = BenchmarkResult(
            model="hrm",
            tokens_generated=100,
            total_time_s=10.0,
        )
        assert r.tokens_per_sec == 0.0  # not auto-computed

    def test_tokens_per_sec_manual(self) -> None:
        r = BenchmarkResult(
            model="fused",
            tokens_generated=50,
            total_time_s=2.0,
            tokens_per_sec=25.0,
        )
        assert r.tokens_per_sec == 25.0


class TestFormatTable:
    def test_empty(self) -> None:
        assert "Config" in format_table([])

    def test_single_row(self) -> None:
        r = BenchmarkResult(
            model="ouro",
            tokens_generated=42,
            total_time_s=2.0,
            tokens_per_sec=21.0,
            ttft_s=0.5,
            memory_mb=1024.0,
        )
        table = format_table([r])
        assert "ouro/average" in table
        assert "21.0" in table

    def test_multiple_rows(self) -> None:
        results = [
            BenchmarkResult(
                model="ouro", tokens_generated=50, total_time_s=2.0, tokens_per_sec=25.0
            ),
            BenchmarkResult(
                model="hrm", tokens_generated=40, total_time_s=2.0, tokens_per_sec=20.0
            ),
        ]
        table = format_table(results)
        assert "ouro/average" in table
        assert "hrm/average" in table
        assert "25.0" in table
        assert "20.0" in table


class TestMemory:
    def test_maybe_get_memory_returns_float(self) -> None:
        mem = maybe_get_memory_mb()
        assert isinstance(mem, float)
        assert mem >= 0.0


class TestRobustnessBattery:
    def test_battery_is_populated(self) -> None:
        assert len(ROBUSTNESS_BATTERY) >= 20

    def test_all_prompts_have_category(self) -> None:
        for entry in ROBUSTNESS_BATTERY:
            assert "prompt" in entry
            assert "category" in entry
            assert entry["prompt"]

    def test_categories_are_diverse(self) -> None:
        cats = {e["category"] for e in ROBUSTNESS_BATTERY}
        assert "factual" in cats
        assert "reasoning" in cats
        assert "math" in cats
        assert "creative" in cats

    def test_no_duplicate_prompts(self) -> None:
        prompts = [e["prompt"] for e in ROBUSTNESS_BATTERY]
        assert len(prompts) == len(set(prompts))


class TestRobustnessResult:
    def test_defaults(self) -> None:
        r = RobustnessResult()
        assert r.category == ""
        assert r.ouro_ppl == 0.0
        assert r.fusion_win_rate == 0.0

    def test_custom_values(self) -> None:
        r = RobustnessResult(
            prompt="test",
            category="math",
            avg_fusion_gain=0.5,
            fusion_win_rate=0.8,
        )
        assert r.avg_fusion_gain == 0.5
        assert r.fusion_win_rate == 0.8


class TestFormatRobustnessTable:
    def test_empty(self) -> None:
        table = format_robustness_table([])
        assert "no results" in table

    def test_single_entry(self) -> None:
        r = RobustnessResult(prompt="test", category="math", avg_fusion_gain=0.5)
        table = format_robustness_table([r])
        assert "math" in table
        assert "TOTAL" in table

    def test_grouped_by_category(self) -> None:
        results = [
            RobustnessResult(prompt="a", category="math", avg_fusion_gain=0.1),
            RobustnessResult(prompt="b", category="math", avg_fusion_gain=0.2),
            RobustnessResult(prompt="c", category="code", avg_fusion_gain=0.3),
        ]
        table = format_robustness_table(results)
        assert "math" in table
        assert "code" in table
        assert "TOTAL" in table

    def test_fusion_verdict_positive(self) -> None:
        r = RobustnessResult(avg_fusion_gain=0.1)
        table = format_robustness_table([r])
        assert "YES" in table

    def test_fusion_verdict_negative(self) -> None:
        r = RobustnessResult(avg_fusion_gain=-0.1)
        table = format_robustness_table([r])
        assert "NO" in table
