#!/usr/bin/env python3
"""Autoregressive text completion using fused Ouro-1.4B + HRM-Text-1B.

HRM requires chat format (<|im_start|><|condition|>prompt<|im_end|>),
condition tags (direct/cot/noisy/synth), and prefix-LM token_type_ids.

Usage:
  python3 text_completion.py --local "The capital of France is"
  python3 text_completion.py --model ouro --local "The capital of France is"
  python3 text_completion.py --model hrm --local -n 30 "The capital of France is"
  python3 text_completion.py --model hrm --condition cot --local "Explain the sky"
  python3 text_completion.py --local -n 30 --rep-penalty 1.1 --ouro-weight 0.3 "Q: What is"
"""

import argparse
import math
import sys
from pathlib import Path
from tokenizers import Tokenizer

from token_matcher import TokenMatcher

BASE = Path(__file__).parent.resolve()

HRM_EOS_ID = 11
OURO_EOS_ID = 0


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

    def sample_token(self, ouro_logits: list[float], hrm_logits: list[float], temperature: float = 1.0) -> tuple[int, str, float]:
        import random
        candidates = self.fuse_logits(ouro_logits, hrm_logits)
        if not candidates:
            return 0, "", 0.0
        if temperature <= 0 or len(candidates) == 1:
            return candidates[0][0], candidates[0][2], candidates[0][1]
        probs = [p for _, p, _ in candidates]
        temp_probs = [math.log(max(p, 1e-10)) / temperature for p in probs]
        max_log = max(temp_probs)
        weights = [math.exp(lp - max_log) for lp in temp_probs]
        total = sum(weights)
        normalized = [w / total for w in weights]
        r = random.random()
        cumulative = 0.0
        for i, w in enumerate(normalized):
            cumulative += w
            if r <= cumulative:
                return candidates[i][0], candidates[i][2], candidates[i][1]
        return candidates[-1][0], candidates[-1][2], candidates[-1][1]


def patch_ouro_model(config):
    config._attn_implementation = "eager"


def softmax(logits: list[float], k: int) -> tuple[list[int], list[float]]:
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])[:k]
    top_ids = [i for i, _ in indexed]
    top_vals = [v for _, v in indexed]
    max_val = max(top_vals)
    exps = [math.exp(v - max_val) for v in top_vals]
    total = sum(exps)
    probs = [e / total for e in exps]
    return top_ids, probs


def sample_from_logits(logits: list[float], tok: Tokenizer, k: int, temperature: float) -> tuple[int, str, float]:
    import random
    ids, probs = softmax(logits, k)
    if temperature <= 0 or len(ids) == 1:
        return ids[0], tok.decode([ids[0]]), probs[0]
    temp_probs = [math.log(max(p, 1e-10)) / temperature for p in probs]
    max_log = max(temp_probs)
    weights = [math.exp(lp - max_log) for lp in temp_probs]
    total = sum(weights)
    normalized = [w / total for w in weights]
    r = random.random()
    cumulative = 0.0
    for i, w in enumerate(normalized):
        cumulative += w
        if r <= cumulative:
            return ids[i], tok.decode([ids[i]]), probs[i]
    return ids[-1], tok.decode([ids[-1]]), probs[-1]


def apply_repetition_penalty(logits: list[float], seen_ids: set[int], penalty: float) -> list[float]:
    if penalty == 1.0 or not seen_ids:
        return logits
    out = list(logits)
    for tid in seen_ids:
        if 0 <= tid < len(out):
            out[tid] /= penalty if out[tid] >= 0 else (2 - penalty)
    return out


def format_hrm_prompt(text: str, condition: str) -> str:
    return f"<|im_start|><|{condition}|>{text}<|im_end|>"


def strip_hrm_output(text: str) -> str:
    import re
    text = re.sub(r'<\|im_start\|>.*?<\|im_end\|>', '', text)
    text = text.replace('<|box_end|>', '').replace('<|box_start|>', '')
    return text.strip()


def generate(
    text: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    threshold: float,
    ouro_weight: float,
    local: bool,
    model: str = "fused",
    repetition_penalty: float = 1.0,
    condition: str = "direct",
):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("Error: requires torch and transformers", file=sys.stderr)
        sys.exit(1)

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

    load_ouro = model in ("fused", "ouro")
    load_hrm = model in ("fused", "hrm")

    print(f"Loading models on {device}...", file=sys.stderr)
    if load_ouro:
        from transformers import AutoConfig
        ouro_config = AutoConfig.from_pretrained(ouro_path, trust_remote_code=True)
        patch_ouro_model(ouro_config)
        ouro_model = AutoModelForCausalLM.from_pretrained(
            ouro_path, config=ouro_config, torch_dtype=dtype, device_map=device,
            trust_remote_code=True,
        )
        if device == "cpu":
            from ouro_cache_fix import UniversalTransformerCache
            ouro_cache = UniversalTransformerCache()
        else:
            ouro_cache = None
    if load_hrm:
        hrm_model = AutoModelForCausalLM.from_pretrained(
            hrm_path, torch_dtype=dtype, device_map=device,
        )

    fuser = Fuser(matcher, ouro_tok, hrm_tok, ouro_weight, top_k, threshold)

    label = {"fused": "Fused", "ouro": "Ouro-1.4B", "hrm": "HRM-Text-1B"}[model]
    print(f"Model: {label}")
    if model == "fused":
        print(f"Weights: Ouro={ouro_weight}  HRM={1-ouro_weight}")
    print(f"Generating up to {max_new_tokens} tokens (cond={condition})")
    print("-" * 60)

    # Encode prompt — Ouro gets raw text, HRM gets chat format
    if load_ouro:
        ouro_ids = ouro_tok.encode(text).ids
        ouro_gen_ids: set[int] = set()
    if load_hrm:
        hrm_prompt = format_hrm_prompt(text, condition)
        hrm_ids = hrm_tok.encode(hrm_prompt).ids
        hrm_gen_ids: set[int] = set()

    print(f"Prompt (Ouro: {len(ouro_ids) if load_ouro else 0} tok, HRM: {len(hrm_ids) if load_hrm else 0} tok)")
    print(text)
    print("-" * 60)

    for step in range(max_new_tokens):
        if load_ouro:
            with torch.no_grad():
                ouro_kwargs = {}
                if device == "cpu" and step > 0:
                    ouro_kwargs["past_key_values"] = ouro_cache
                    ouro_kwargs["use_cache"] = True
                ouro_out = ouro_model(
                    input_ids=torch.tensor([ouro_ids], device=device),
                    **ouro_kwargs,
                )
            ouro_logits = ouro_out.logits[0, -1, :].tolist()
            if repetition_penalty != 1.0:
                ouro_logits = apply_repetition_penalty(ouro_logits, ouro_gen_ids, repetition_penalty)

        if load_hrm:
            hrm_tti = torch.ones(len(hrm_ids), dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad():
                hrm_out = hrm_model(input_ids=torch.tensor([hrm_ids], device=device), token_type_ids=hrm_tti)
            hrm_logits = hrm_out.logits[0, -1, :].tolist()
            if repetition_penalty != 1.0:
                hrm_logits = apply_repetition_penalty(hrm_logits, hrm_gen_ids, repetition_penalty)

        if model == "fused":
            tid, token_str, prob = fuser.sample_token(ouro_logits, hrm_logits, temperature)
            hrm_ids.append(tid)
            hrm_gen_ids.add(tid)
            m = matcher.hrm_to_ouro(tid)
            if m.target_ids:
                ouro_ids.extend(m.target_ids)
                for otid in m.target_ids:
                    ouro_gen_ids.add(otid)
            else:
                ouro_ids.extend(ouro_tok.encode(token_str).ids)
            eos_id = HRM_EOS_ID
        elif model == "ouro":
            tid, token_str, prob = sample_from_logits(ouro_logits, ouro_tok, top_k, temperature)
            ouro_ids.append(tid)
            ouro_gen_ids.add(tid)
            eos_id = OURO_EOS_ID
        elif model == "hrm":
            tid, token_str, prob = sample_from_logits(hrm_logits, hrm_tok, top_k, temperature)
            hrm_ids.append(tid)
            hrm_gen_ids.add(tid)
            eos_id = HRM_EOS_ID

        if tid == eos_id:
            print(f"\n[EOS at step {step + 1}]")
            break

        if token_str:
            print(token_str, end="", flush=True)
        else:
            print(f"[tok {tid}]", end="", flush=True)

    print()
    print("-" * 60)
    print(f"Generated {step + 1} tokens")
    return


def main():
    parser = argparse.ArgumentParser(description="Fused Ouro+HRM text completion")
    parser.add_argument("prompt", nargs="?", help="Input text prompt")
    parser.add_argument("--local", action="store_true", help="Load models from local dirs")
    parser.add_argument("-n", "--max-new-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temp", "--temperature", type=float, default=1.0, dest="temperature",
                        help="Sampling temperature (0=greedy)")
    parser.add_argument("--top-k", type=int, default=30, help="Top-k tokens per model")
    parser.add_argument("--threshold", type=float, default=0.01, help="Min probability")
    parser.add_argument("--ouro-weight", type=float, default=0.5, help="Ouro fusion weight")
    parser.add_argument("--model", "--m", choices=["fused", "ouro", "hrm"], default="fused",
                        help="Which model to use: fused, ouro, or hrm (default: fused)")
    parser.add_argument("--rep-penalty", type=float, default=1.0, dest="repetition_penalty",
                        help="Repetition penalty (>1.0 discourages repeats, default=1.0)")
    parser.add_argument("--condition", choices=["direct", "cot", "noisy", "synth"], default="direct",
                        help="HRM condition tag (direct/cot/noisy/synth, default: direct)")

    args = parser.parse_args()
    if not args.prompt:
        parser.print_help()
        sys.exit(1)

    generate(
        text=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        threshold=args.threshold,
        ouro_weight=args.ouro_weight,
        local=args.local,
        model=args.model,
        repetition_penalty=args.repetition_penalty,
        condition=args.condition,
    )


if __name__ == "__main__":
    main()
