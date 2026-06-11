"""CLI entry point: python -m llm_fusion."""

import sys

from llm_fusion.benchmark import main as bench_main
from llm_fusion.cli import main as cli_main


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "benchmark":
        sys.argv.pop(1)
        return bench_main()
    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
