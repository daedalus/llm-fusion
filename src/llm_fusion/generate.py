"""Autoregressive text completion using fused Ouro-1.4B + HRM-Text-1B."""

from __future__ import annotations

import logging
import math
import random
import re
import sys
from pathlib import Path

from tokenizers import Tokenizer

from llm_fusion.fusion import Fuser, compute_kl, softmax_top_k
from llm_fusion.metrics import fusion_gain as _compute_gain
from llm_fusion.token_matcher import TokenMatcher

HRM_EOS_ID = 11
OURO_EOS_ID = 0

try:
    from ouro_cache_fix import UniversalTransformerCache  # noqa: F401

    HAS_OURO_CACHE = True
except ImportError:
    HAS_OURO_CACHE = False


def format_hrm_prompt(text: str, condition: str) -> str:
    return f"<|im_start|><|{condition}|>{text}<|im_end|>"


def strip_hrm_output(text: str) -> str:
    text = re.sub(r"<\|im_start\|>.*?<\|im_end\|>", "", text)
    text = text.replace("<|box_end|>", "").replace("<|box_start|>", "")
    return text.strip()


def apply_repetition_penalty(
    logits: list[float],
    seen_ids: set[int],
    penalty: float,
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
    logits: list[float],
    tok: Tokenizer,
    k: int,
    temperature: float,
    rng: random.Random | None = None,
) -> tuple[int, str, float]:
    if rng is None:
        rng = random.Random()
    ids, probs = softmax_top_k(logits, k)
    if temperature <= 0 or len(ids) == 1:
        return ids[0], tok.decode([ids[0]]), probs[0]
    temp_probs = [math.log(max(p, 1e-10)) / temperature for p in probs]
    max_log = max(temp_probs)
    weights = [math.exp(lp - max_log) for lp in temp_probs]
    total = sum(weights)
    normalized = [w / total for w in weights]
    r = rng.random()
    cumulative = 0.0
    for i, w in enumerate(normalized):
        cumulative += w
        if r <= cumulative:
            return ids[i], tok.decode([ids[i]]), probs[i]
    return ids[-1], tok.decode([ids[-1]]), probs[-1]


def compute_perplexity(
    text: str,
    model: CausalLM,
    tokenizer: Tokenizer,
    device: str = "cpu",
    stride: int = 512,
) -> float:
    import torch

    input_ids = tokenizer.encode(text).ids
    if not input_ids:
        return float("inf")
    nll = 0.0
    n_tokens = 0
    seq_len = len(input_ids)
    for start in range(0, seq_len - 1, stride):
        end = min(start + stride, seq_len)
        chunk = input_ids[max(0, start - 1) : end] if start > 0 else input_ids[:end]
        inp = torch.tensor([chunk], device=device)
        with torch.no_grad():
            out = model(inp)
        logits = out.logits[0]
        shift_logits = logits[:-1]
        shift_labels = inp[0, 1:]
        nll += (
            torch.nn.functional.cross_entropy(
                shift_logits,
                shift_labels,
                reduction="none",
            )
            .sum()
            .item()
        )
        n_tokens += len(shift_labels)
    return math.exp(nll / max(n_tokens, 1))


def compute_fused_perplexity(
    text: str,
    ouro_model: CausalLM,
    hrm_model: CausalLM,
    ouro_tok: Tokenizer,
    hrm_tok: Tokenizer,
    fuser: Fuser,
    device: str = "cpu",
) -> float:
    import torch

    hrm_ids = hrm_tok.encode(text).ids
    if len(hrm_ids) < 2:
        return float("inf")
    nll = 0.0
    ouro_tok.encode(text).ids
    for t in range(1, len(hrm_ids)):
        ouro_prefix = ouro_tok.encode(hrm_tok.decode(hrm_ids[:t])).ids or [0]
        inp = torch.tensor([ouro_prefix], device=device)
        with torch.no_grad():
            ouro_out = ouro_model(inp)
        ouro_logits = ouro_out.logits[0, -1, :].tolist()
        hrm_inp = torch.tensor([hrm_ids[:t]], device=device)
        hrm_tti = torch.ones(t, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            hrm_out = hrm_model(hrm_inp, token_type_ids=hrm_tti)
        hrm_logits = hrm_out.logits[0, -1, :].tolist()
        candidates = fuser.fuse_logits(ouro_logits, hrm_logits)
        target_tid = hrm_ids[t]
        fused_prob = 0.0
        for tid, p, _ in candidates:
            if tid == target_tid:
                fused_prob = p
                break
        nll -= math.log(max(fused_prob, 1e-10))
    return math.exp(nll / (len(hrm_ids) - 1))


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
    strategy: str = "dynamic",
    cascade_threshold: float = 0.5,
    dynamic_initial_weight: float = 0.8,
    dynamic_final_weight: float = 0.2,
    perplexity: bool = False,
    show_kl: bool = False,
    show_gain: bool = False,
    eval_text: str = "",
    verbose: bool = False,
    debug: bool = False,
    seed: int | None = None,
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

    rng = random.Random(seed)

    bd = Path(base_dir) if base_dir else Path(__file__).resolve().parent.parent.parent
    ouro_tok_path = bd / "Ouro-1.4B/tokenizer.json"
    hrm_tok_path = bd / "HRM-Text-1B/tokenizer.json"

    log = logging.getLogger(__name__)
    if debug:
        log.setLevel(logging.DEBUG)
    elif verbose:
        log.setLevel(logging.INFO)

    log.info("Initializing token matcher from %s and %s", ouro_tok_path, hrm_tok_path)
    matcher = TokenMatcher(ouro_tok_path, hrm_tok_path)
    ouro_tok = Tokenizer.from_file(str(ouro_tok_path))
    hrm_tok = Tokenizer.from_file(str(hrm_tok_path))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cpu" else torch.float16
    load_ouro = model in ("fused", "ouro")
    load_hrm = model in ("fused", "hrm")

    log.info("Device: %s, dtype: %s", device, dtype)
    log.debug("load_ouro=%s, load_hrm=%s", load_ouro, load_hrm)

    print(f"Loading models on {device}...", file=sys.stderr)
    if load_ouro:
        if local:
            ouro_model_path = str(bd / "Ouro-1.4B")
        else:
            ouro_model_path = ouro_path
        log.info("Loading Ouro model from %s", ouro_model_path)
        ouro_config = AutoConfig.from_pretrained(ouro_model_path, trust_remote_code=True)
        patch_ouro_model(ouro_config)
        log.debug("Ouro config._attn_implementation set to 'eager'")
        ouro_model = AutoModelForCausalLM.from_pretrained(
            ouro_model_path,
            config=ouro_config,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
        log.info("Ouro model loaded")
    if load_hrm:
        if local:
            hrm_model_path = str(bd / "HRM-Text-1B")
        else:
            hrm_model_path = hrm_path
        log.info("Loading HRM model from %s", hrm_model_path)
        hrm_model = AutoModelForCausalLM.from_pretrained(
            hrm_model_path,
            torch_dtype=dtype,
            device_map=device,
        )
        log.info("HRM model loaded")

    if perplexity:
        print(f"Computing perplexity for {model}...")
        print("-" * 60)
        if model == "fused":
            fuser = Fuser(
                matcher,
                ouro_tok,
                hrm_tok,
                ouro_weight,
                top_k,
                threshold,
                strategy,
                cascade_threshold,
                dynamic_initial_weight,
                dynamic_final_weight,
                0,
            )
            ppl = compute_fused_perplexity(
                text, ouro_model, hrm_model, ouro_tok, hrm_tok, fuser, device
            )
        else:
            model_obj = ouro_model if load_ouro else hrm_model
            tok = ouro_tok if load_ouro else hrm_tok
            ppl = compute_perplexity(text, model_obj, tok, device)
        print(f"Perplexity: {ppl:.2f}")
        print("-" * 60)
        return

    if eval_text:
        print(f"Evaluating on {len(eval_text)} chars...")
        print("-" * 60)
        from llm_fusion.metrics import evaluate_text as _eval_fn

        results = _eval_fn(
            eval_text,
            ouro_model,
            hrm_model,
            ouro_tok,
            hrm_tok,
            fuser if model == "fused" else None,
            device,
            max_new_tokens,
        )
        print(f"  Tokens evaluated:   {results['n_tokens']}")
        print(
            f"  Avg fusion gain:    {results['avg_fusion_gain']:+.4f}  (log-ratio vs best parent)"
        )
        print(f"  Fusion win rate:    {results['fusion_win_rate']:.1%}  (fusion beats best parent)")
        print(f"  Oracle agreement:   {results['oracle_rate']:.1%}  (agreement with better parent)")
        print(f"  Ouro PPL:           {results['ouro_ppl']:.2f}")
        print(f"  HRM PPL:            {results['hrm_ppl']:.2f}")
        print(f"  Fused PPL:          {results['fused_ppl']:.2f}")
        print(f"  PPL vs Ouro:        {results['ppl_improvement_vs_ouro']:+.1f}%")
        print(f"  PPL vs HRM:         {results['ppl_improvement_vs_hrm']:+.1f}%")
        print("-" * 60)
        return

    fuser = Fuser(
        matcher,
        ouro_tok,
        hrm_tok,
        ouro_weight,
        top_k,
        threshold,
        strategy,
        cascade_threshold,
        dynamic_initial_weight,
        dynamic_final_weight,
        max_new_tokens,
    )
    label = {"fused": "Fused", "ouro": "Ouro-1.4B", "hrm": "HRM-Text-1B"}[model]
    print(f"Model: {label}")
    if model == "fused":
        print(f"Strategy: {strategy}")
        if strategy == "average":
            print(f"Weights: Ouro={ouro_weight}  HRM={1 - ouro_weight}")
        elif strategy == "cascade":
            print(f"Cascade threshold: {cascade_threshold}")
        elif strategy == "dynamic":
            print(
                f"Ouro weight: {dynamic_initial_weight} -> {dynamic_final_weight} over {max_new_tokens} steps"
            )
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
    prev_gen_text = ""

    print(
        f"Prompt (Ouro: {len(ouro_prompt_ids) if load_ouro else 0} tok, "
        f"HRM: {len(hrm_ids) if load_hrm else 0} tok)"
    )
    print(text)
    print("-" * 60)

    ouro_cache = None
    if HAS_OURO_CACHE and model != "fused" and load_ouro:
        ouro_cache = UniversalTransformerCache()

    for step in range(max_new_tokens):
        log.debug("Step %d, generated_text length %d, generated_text=%s", step, len(generated_text), repr(generated_text[-40:]))
        if load_ouro:
            if model == "fused":
                new_part = generated_text[len(prev_gen_text):]
                if not prev_gen_text:
                    ouro_ids = ouro_prompt_ids + ouro_tok.encode(generated_text).ids
                else:
                    overlap = min(len(prev_gen_text), 32)
                    resuffix = prev_gen_text[-overlap:] + new_part
                    resuffix_ids = ouro_tok.encode(resuffix).ids
                    keep = len(ouro_ids) - overlap
                    ouro_ids = ouro_ids[:keep] + resuffix_ids
                prev_gen_text = generated_text
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
                ouro_logits = apply_repetition_penalty(
                    ouro_logits, ouro_gen_ids, repetition_penalty
                )

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
            fuser.current_step = step
            tid, token_str, prob = fuser.sample_token(ouro_logits, hrm_logits, temperature, rng)
            if show_kl:
                ouro_dist, hrm_dist = fuser.model_distributions(ouro_logits, hrm_logits)
                kl_oh = compute_kl(ouro_dist, hrm_dist)
                kl_ho = compute_kl(hrm_dist, ouro_dist)
                print(f" [KL o→h={kl_oh:.2f} h→o={kl_ho:.2f}]", end="", flush=True)
            if show_gain:
                ouro_probs = dict(zip(*softmax_top_k(ouro_logits, 50)))
                hrm_probs = dict(zip(*softmax_top_k(hrm_logits, 50)))
                ouro_p = ouro_probs.get(tid, 0.0)
                hrm_p = hrm_probs.get(tid, 0.0)
                gain = _compute_gain(prob, ouro_p, hrm_p)
                if gain > 0:
                    print(f" [gain=+{gain:.3f}]", end="", flush=True)
                else:
                    print(f" [gain={gain:.3f}]", end="", flush=True)
            hrm_ids.append(tid)
            hrm_gen_ids.add(tid)
            eos_id = HRM_EOS_ID
        elif model == "ouro":
            tid, token_str, prob = sample_from_logits(ouro_logits, ouro_tok, top_k, temperature, rng)
            ouro_ids.append(tid)
            ouro_gen_ids.add(tid)
            eos_id = OURO_EOS_ID
        elif model == "hrm":
            tid, token_str, prob = sample_from_logits(hrm_logits, hrm_tok, top_k, temperature, rng)
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
    log.info("Generated %d tokens with model=%s strategy=%s", step + 1, model, strategy)
