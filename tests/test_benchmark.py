"""Tests for llm_fusion.benchmark."""

from __future__ import annotations

from llm_fusion.benchmark import BenchmarkResult, format_table, maybe_get_memory_mb


class TestBenchmarkResult:
    def test_defaults(self):
        r = BenchmarkResult(model="ouro")
        assert r.model == "ouro"
        assert r.tokens_per_sec == 0.0

    def test_tokens_per_sec_computed(self):
        r = BenchmarkResult(
            model="hrm", tokens_generated=100, total_time_s=10.0,
        )
        assert r.tokens_per_sec == 0.0  # not auto-computed

    def test_tokens_per_sec_manual(self):
        r = BenchmarkResult(
            model="fused", tokens_generated=50, total_time_s=2.0,
            tokens_per_sec=25.0,
        )
        assert r.tokens_per_sec == 25.0


class TestFormatTable:
    def test_empty(self):
        assert "Config" in format_table([])

    def test_single_row(self):
        r = BenchmarkResult(model="ouro", tokens_generated=42, total_time_s=2.0,
                            tokens_per_sec=21.0, ttft_s=0.5, memory_mb=1024.0)
        table = format_table([r])
        assert "ouro/average" in table
        assert "21.0" in table

    def test_multiple_rows(self):
        results = [
            BenchmarkResult(model="ouro", tokens_generated=50, total_time_s=2.0,
                            tokens_per_sec=25.0),
            BenchmarkResult(model="hrm", tokens_generated=40, total_time_s=2.0,
                            tokens_per_sec=20.0),
        ]
        table = format_table(results)
        assert "ouro/average" in table
        assert "hrm/average" in table
        assert "25.0" in table
        assert "20.0" in table


class TestMemory:
    def test_maybe_get_memory_returns_float(self):
        mem = maybe_get_memory_mb()
        assert isinstance(mem, float)
        assert mem >= 0.0
