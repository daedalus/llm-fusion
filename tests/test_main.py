"""Tests for llm_fusion.__main__."""

from __future__ import annotations

import sys

import pytest


class TestMain:
    def test_main_no_args_returns_1(self) -> None:
        from llm_fusion.__main__ import main

        old_argv = sys.argv[:]
        sys.argv = ["llm-fusion"]
        try:
            result = main()
            assert result == 1
        finally:
            sys.argv = old_argv

    def test_main_benchmark_dispatches(self) -> None:
        from llm_fusion.__main__ import main

        old_argv = sys.argv[:]
        sys.argv = ["llm-fusion", "benchmark", "--help"]
        try:
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
        finally:
            sys.argv = old_argv
