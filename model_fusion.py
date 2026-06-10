#!/usr/bin/env python3
"""Fuse token probability distributions from Ouro-1.4B and HRM-Text-1B.

Both models predict the next token given the same prompt. Ouro's output
distribution is mapped into HRM's vocabulary space via TokenMatcher, then
the two distributions are fused with a weighted average. Tokens above a
probability threshold are decoded with HRM's tokenizer.

Usage:
  python3 model_fusion.py --local "The capital of France is"
  python3 model_fusion.py --top-k 30 --threshold 0.05 --ouro-weight 0.4 "prompt"
  python3 model_fusion.py --demo "lorem ipsum"
"""

import argparse
import math
import sys
from pathlib import Path
from tokenizers import Tokenizer

from token_matcher import TokenMatcher

BASE = Path(__file__).parent.resolve()


def softmax_top_k(logits: list[float], k: int) -> tuple[list[int], list[float]]:
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])[:k]
    top_ids = [i for i, _ in indexed]
    top_vals = [v for _, v in indexed]
    max_val = max(top_vals)
    exps = [math.exp(v - max_val) for v in top_vals]
    total = sum(exps)
    probs = [e / total for e in exps]
    return top_ids, probs


class Fuser:
    def __init__(
        self,
        matcher: TokenMatcher,
        ouro_tok: Tokenizer,
        hrm_tok: Tokenizer,
        ouro_weight: float = 0.5,
        top_k: int = 50,
        threshold: float = 0.01,
    ):
        self.matcher = matcher
        self.ouro_tok = ouro_tok
        self.hrm_tok = hrm_tok
        self.ouro_weight = ouro_weight
        self.hrm_weight = 1.0 - ouro_weight
        self.top_k = top_k
        self.threshold = threshold

    def fuse_logits(self, ouro_logits: list[float], hrm_logits: list[float]) -> list[tuple[int, float, str]]:
        ouro_top_ids, ouro_probs = softmax_top_k(ouro_logits, self.top_k)
        hrm_top_ids, hrm_probs = softmax_top_k(hrm_logits, self.top_k)

        fused: dict[int, float] = {}

        for tid, prob in zip(hrm_top_ids, hrm_probs):
            fused[tid] = fused.get(tid, 0.0) + prob * self.hrm_weight

        for oid, prob in zip(ouro_top_ids, ouro_probs):
            match = self.matcher.ouro_to_hrm(oid)
            if not match.target_ids:
                continue
            share = prob / len(match.target_ids)
            for tid in match.target_ids:
                fused[tid] = fused.get(tid, 0.0) + share * self.ouro_weight

        filtered = [(tid, p) for tid, p in fused.items() if p >= self.threshold]
        filtered.sort(key=lambda x: -x[1])

        return [(tid, p, self.hrm_tok.decode([tid])) for tid, p in filtered]


def patch_ouro_rope():
    import torch
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    if "default" not in ROPE_INIT_FUNCTIONS:
        def default_rope_init(config, device=None, **kw):
            with torch.no_grad():
                theta = getattr(config, "rope_theta", 10000.0)
                dim = config.hidden_size // config.num_attention_heads
                base = theta ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim)
                inv_freq = 1.0 / base
            return inv_freq, 1.0
        ROPE_INIT_FUNCTIONS["default"] = default_rope_init


def run_inference(
    text: str, top_k: int, threshold: float, ouro_weight: float, local: bool
):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("Error: requires torch and transformers", file=sys.stderr)
        sys.exit(1)

    patch_ouro_rope()

    matcher = TokenMatcher()
    ouro_tok = Tokenizer.from_file(str(BASE / "Ouro-1.4B/tokenizer.json"))
    hrm_tok = Tokenizer.from_file(str(BASE / "HRM-Text-1B/tokenizer.json"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    if local:
        ouro_path = str(BASE / "Ouro-1.4B")
        hrm_path = str(BASE / "HRM-Text-1B")
    else:
        ouro_path = "ByteDance/Ouro-1.4B"
        hrm_path = "sapientinc/HRM-Text-1B"

    print(f"Loading models on {device}...", file=sys.stderr)
    ouro_model = AutoModelForCausalLM.from_pretrained(
        ouro_path, torch_dtype=dtype, device_map=device, trust_remote_code=True
    )
    hrm_model = AutoModelForCausalLM.from_pretrained(
        hrm_path, torch_dtype=dtype, device_map=device,
    )

    ouro_ids = ouro_tok.encode(text).ids
    hrm_ids = hrm_tok.encode(text).ids

    with torch.no_grad():
        ouro_out = ouro_model(input_ids=torch.tensor([ouro_ids], device=device))
        hrm_out = hrm_model(input_ids=torch.tensor([hrm_ids], device=device))

    ouro_logits = ouro_out.logits[0, -1, :].tolist()
    hrm_logits = hrm_out.logits[0, -1, :].tolist()

    fuser = Fuser(matcher, ouro_tok, hrm_tok, ouro_weight, top_k, threshold)
    results = fuser.fuse_logits(ouro_logits, hrm_logits)

    print(f"Prompt: {text!r}")
    print(f"Ouro tokens: {len(ouro_ids)}  HRM tokens: {len(hrm_ids)}")
    print(f"Weights: Ouro={ouro_weight}  HRM={1-ouro_weight}")
    print()

    ouro_top5 = sorted(enumerate(ouro_logits), key=lambda x: -x[1])[:5]
    hrm_top5 = sorted(enumerate(hrm_logits), key=lambda x: -x[1])[:5]
    print(f"Ouro top-5: {[ouro_tok.decode([i]) for i, _ in ouro_top5]}")
    print(f"HRM  top-5: {[hrm_tok.decode([i]) for i, _ in hrm_top5]}")
    print()

    print(f"Fused (threshold={threshold}, top {len(results)} total):")
    print(f"  {'HRM ID':>7}  {'Prob':>7}  {'Token':<20}  {'Source contribution'}")
    print(f"  {'-'*55}")
    for tid, prob, token_str in results[:12]:
        ouro_contrib = sum(
            prob * ouro_weight / len(matcher.ouro_to_hrm(oid).target_ids)
            for oid, _ in ouro_top5
            if tid in matcher.ouro_to_hrm(oid).target_ids
        ) / sum(p for _, p in ouro_top5) if ouro_top5 else 0
        src = "both" if ouro_contrib > 0.001 and prob * (1 - ouro_weight) > 0.001 else \
              "ouro" if ouro_contrib > 0.001 else "hrm"
        print(f"  [{tid:>5}]  {prob:.4f}  {token_str!r:<20}  {src}")

    print()
    decoded_tokens = [s for _, _, s in results]
    print(f"Fused top tokens: {' + '.join(decoded_tokens[:5])}")


def run_demo(text: str, top_k: int, threshold: float, ouro_weight: float):
    matcher = TokenMatcher()
    ouro_tok = Tokenizer.from_file(str(BASE / "Ouro-1.4B/tokenizer.json"))
    hrm_tok = Tokenizer.from_file(str(BASE / "HRM-Text-1B/tokenizer.json"))

    fuser = Fuser(matcher, ouro_tok, hrm_tok, ouro_weight, top_k, threshold)

    import random
    rng = random.Random(42)
    ouro_ids = ouro_tok.encode(text).ids
    hrm_ids = hrm_tok.encode(text).ids
    ouro_logits = [rng.gauss(0, 1) for _ in range(ouro_tok.get_vocab_size())]
    hrm_logits = [rng.gauss(0, 1) for _ in range(hrm_tok.get_vocab_size())]

    print("=" * 60)
    print("FUSION DEMO (synthetic logits)")
    print("=" * 60)
    print(f"Prompt: {text!r}")
    print(f"Ouro: {len(ouro_ids)} tokens  HRM: {len(hrm_ids)} tokens")
    print(f"Config: top_k={top_k}, threshold={threshold}, ouro_weight={ouro_weight}")
    print()

    results = fuser.fuse_logits(ouro_logits, hrm_logits)
    print(f"Fused tokens (above {threshold}): {len(results)}")
    for tid, prob, s in results[:15]:
        print(f"  [{tid:>5}] p={prob:.4f}  {s!r}")


def main():
    parser = argparse.ArgumentParser(description="Fuse Ouro + HRM token predictions")
    parser.add_argument("prompt", nargs="?", help="Input text prompt")
    parser.add_argument("--demo", action="store_true", help="Run with synthetic logits")
    parser.add_argument("--local", action="store_true", help="Load models from local dirs")
    parser.add_argument("--top-k", type=int, default=30, help="Top-k tokens per model")
    parser.add_argument("--threshold", type=float, default=0.01, help="Min probability")
    parser.add_argument("--ouro-weight", type=float, default=0.5, help="Ouro fusion weight")

    args = parser.parse_args()
    if not args.prompt:
        parser.print_help()
        sys.exit(1)

    if args.demo:
        run_demo(args.prompt, args.top_k, args.threshold, args.ouro_weight)
    else:
        run_inference(args.prompt, args.top_k, args.threshold, args.ouro_weight, args.local)


if __name__ == "__main__":
    main()
