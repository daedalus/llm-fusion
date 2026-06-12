"""CLI entry point for LLM Fusion."""

from __future__ import annotations

import argparse
import logging
import sys

from llm_fusion.generate import generate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fused Ouro-1.4B + HRM-Text-1B text completion",
    )
    parser.add_argument("prompt", nargs="?", help="Input text prompt")
    parser.add_argument("--local", action="store_true", help="Load models from local dirs")
    parser.add_argument(
        "-n",
        "--max-new-tokens",
        type=int,
        default=100,
        help="Max tokens to generate",
    )
    parser.add_argument(
        "--temp",
        "--temperature",
        type=float,
        default=1.0,
        dest="temperature",
        help="Sampling temperature (0=greedy)",
    )
    parser.add_argument("--top-k", type=int, default=30, help="Top-k tokens per model")
    parser.add_argument("--threshold", type=float, default=0.01, help="Min probability")
    parser.add_argument(
        "--ouro-weight",
        type=float,
        default=0.5,
        help="Ouro fusion weight (fused mode only)",
    )
    parser.add_argument(
        "-m",
        "--model",
        "--m",
        choices=["fused", "ouro", "hrm"],
        default="fused",
        help="Which model to use (default: fused)",
    )
    parser.add_argument(
        "--rep-penalty",
        type=float,
        default=1.0,
        dest="repetition_penalty",
        help="Repetition penalty (>1.0 discourages repeats)",
    )
    parser.add_argument(
        "--condition",
        choices=["direct", "cot", "noisy", "synth"],
        default="direct",
        help="HRM condition tag",
    )
    parser.add_argument(
        "--strategy",
        choices=["average", "product", "min-entropy", "cascade", "dynamic"],
        default="dynamic",
        help="Fusion strategy: average, product, min-entropy, cascade, or dynamic",
    )
    parser.add_argument(
        "--cascade-threshold",
        type=float,
        default=0.5,
        help="Ouro top-prob threshold for cascade strategy (default: 0.5)",
    )
    parser.add_argument(
        "--dynamic-initial-weight",
        type=float,
        default=0.8,
        help="Starting Ouro weight for dynamic strategy (default: 0.8)",
    )
    parser.add_argument(
        "--dynamic-final-weight",
        type=float,
        default=0.2,
        help="Final Ouro weight for dynamic strategy (default: 0.2)",
    )
    parser.add_argument(
        "--perplexity",
        action="store_true",
        help="Evaluate perplexity instead of generating",
    )
    parser.add_argument(
        "--kl",
        action="store_true",
        dest="show_kl",
        help="Show per-step KL divergence between model distributions",
    )
    parser.add_argument(
        "--gain",
        action="store_true",
        dest="show_gain",
        help="Show per-step fusion gain (log-ratio vs best parent)",
    )
    parser.add_argument(
        "--eval",
        type=str,
        default="",
        dest="eval_text",
        help="Evaluate fusion against parents on a reference text",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output (INFO logging)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug output (DEBUG logging)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible generation",
    )
    return parser


def setup_logging(verbose: bool = False, debug: bool = False) -> None:
    level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING)
    logging.basicConfig(
        level=level,
        format="%(levelname)-5s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def main() -> int:

    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose, args.debug)

    if not args.prompt:
        parser.print_help()
        return 1

    gen_kwargs = {
        "text": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "threshold": args.threshold,
        "ouro_weight": args.ouro_weight,
        "local": args.local,
        "model": args.model,
        "repetition_penalty": args.repetition_penalty,
        "condition": args.condition,
        "strategy": args.strategy,
        "cascade_threshold": args.cascade_threshold,
        "dynamic_initial_weight": args.dynamic_initial_weight,
        "dynamic_final_weight": args.dynamic_final_weight,
        "perplexity": args.perplexity,
        "show_kl": args.show_kl,
        "show_gain": args.show_gain,
        "eval_text": args.eval_text,
        "verbose": args.verbose,
        "debug": args.debug,
        "seed": args.seed,
    }
    generate(**gen_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
