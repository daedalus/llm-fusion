"""Autoregressive text completion using fused Ouro-1.4B + HRM-Text-1B."""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from tokenizers import Tokenizer

from llm_fusion.token_matcher import TokenMatcher
from llm_fusion.fusion import Fuser, softmax_top_k

HRM_EOS_ID = 11
OURO_EOS_ID = 0

try:
    from ouro_cache_fix import UniversalTransformerCache  # noqa: F401
    HAS_OURO_CACHE = True
except ImportError:
    HAS_OURO_CACHE = False


def patch_ouro_model(config) -> None:
    config._attn_implementation = "eager"


def format_hrm_prompt(text: str, condition: str) -> str:
    return f"<|im_start|><|{condition}|>{text}<|im_end|>"


def strip_hrm_output(text: str) -> str:
    text = re.sub(r"<\|im_start\|>.*?<\|im_end\|>", "", text)
    text = text.replace("<|box_end|>", "").replace("<|box_start|>", "")
    return text.strip()


def apply_repetition_penalty(
    logits: list[float], seen_ids: set[int], penalty: float,
) -> list[float]:
    if penalty == 1.0 or not seen_ids:
        return logits
    out = list(logits)
    for tid in seen_ids:
        if 0 <= tid < len(out):
            divisor = penalty if out[tid] >= 0 else max(2 - penalty, 1e-8)
            out[tid] /= divisor
    return out


def sample_from_logits(
    logits: list[float], tok: Tokenizer, k: int, temperature: float,
) -> tuple[int, str, float]:
    import random
    ids, probs = softmax_top_k(logits, k)
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


def generate(
    text: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 30,
    threshold: float = 0.01,
    ouro_weight: float = 0.5,
    local: bool = False,
    model: str = "fused",
    repetition_penalty: float = 1.0,
    condition: str = "direct",
    strategy: str = "average",
    ouro_path: str = "ByteDance/Ouro-1.4B",
    hrm_path: str = "sapientinc/HRM-Text-1B",
    base_dir: str | Path = "",
) -> None:
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as e:
        print(f"Error: requires torch and transformers ({e})", file=sys.stderr)
        sys.exit(1)

    bd = Path(base_dir) if base_dir else Path(__file__).resolve().parent.parent.parent
    ouro_tok_path = bd / "Ouro-1.4B/tokenizer.json"
    hrm_tok_path = bd / "HRM-Text-1B/tokenizer.json"
    matcher = TokenMatcher(ouro_tok_path, hrm_tok_path)
    ouro_tok = Tokenizer.from_file(str(ouro_tok_path))
    hrm_tok = Tokenizer.from_file(str(hrm_tok_path))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    load_ouro = model in ("fused", "ouro")
    load_hrm = model in ("fused", "hrm")

    print(f"Loading models on {device}...", file=sys.stderr)
    if load_ouro:
        if local:
            ouro_model_path = str(bd / "Ouro-1.4B")
        else:
            ouro_model_path = ouro_path
        ouro_config = AutoConfig.from_pretrained(ouro_model_path, trust_remote_code=True)
        patch_ouro_model(ouro_config)
        ouro_model = AutoModelForCausalLM.from_pretrained(
            ouro_model_path, config=ouro_config, torch_dtype=dtype,
            device_map=device, trust_remote_code=True,
        )
    if load_hrm:
        if local:
            hrm_model_path = str(bd / "HRM-Text-1B")
        else:
            hrm_model_path = hrm_path
        hrm_model = AutoModelForCausalLM.from_pretrained(
            hrm_model_path, torch_dtype=dtype, device_map=device,
        )

    fuser = Fuser(matcher, ouro_tok, hrm_tok, ouro_weight, top_k, threshold, strategy)
    label = {"fused": "Fused", "ouro": "Ouro-1.4B", "hrm": "HRM-Text-1B"}[model]
    print(f"Model: {label}")
    if model == "fused":
        print(f"Weights: Ouro={ouro_weight}  HRM={1-ouro_weight}")
    print(f"Generating up to {max_new_tokens} tokens (cond={condition})")
    print("-" * 60)

    if load_ouro:
        ouro_prompt_ids = ouro_tok.encode(text).ids
    if load_hrm:
        hrm_prompt = format_hrm_prompt(text, condition)
        hrm_ids = hrm_tok.encode(hrm_prompt).ids
        hrm_gen_ids: set[int] = set()

    generated_text = ""
    ouro_gen_ids: set[int] = set()
    ouro_ids = list(ouro_prompt_ids) if load_ouro else []

    print(f"Prompt (Ouro: {len(ouro_prompt_ids) if load_ouro else 0} tok, "
          f"HRM: {len(hrm_ids) if load_hrm else 0} tok)")
    print(text)
    print("-" * 60)

    ouro_cache = None
    if HAS_OURO_CACHE and model != "fused" and load_ouro:
        ouro_cache = UniversalTransformerCache()

    for step in range(max_new_tokens):
        if load_ouro:
            if model == "fused":
                ouro_ids = ouro_prompt_ids + ouro_tok.encode(generated_text).ids
            with torch.no_grad():
                ouro_kwargs = {}
                if ouro_cache is not None and step > 0:
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
                hrm_out = hrm_model(
                    input_ids=torch.tensor([hrm_ids], device=device),
                    token_type_ids=hrm_tti,
                )
            hrm_logits = hrm_out.logits[0, -1, :].tolist()
            if repetition_penalty != 1.0:
                hrm_logits = apply_repetition_penalty(hrm_logits, hrm_gen_ids, repetition_penalty)

        if model == "fused":
            tid, token_str, prob = fuser.sample_token(ouro_logits, hrm_logits, temperature)
            hrm_ids.append(tid)
            hrm_gen_ids.add(tid)
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

        generated_text += token_str
        if token_str:
            print(token_str, end="", flush=True)
        else:
            print(f"[tok {tid}]", end="", flush=True)

    print()
    print("-" * 60)
    print(f"Generated {step + 1} tokens")
