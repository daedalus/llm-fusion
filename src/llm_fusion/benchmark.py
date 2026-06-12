"""Benchmark generation speed, robustness, and fusion quality."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

BENCHMARK_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".benchmark_cache"

try:
    from ouro_cache_fix import UniversalTransformerCache  # noqa: F401

    HAS_OURO_CACHE = True
except ImportError:
    HAS_OURO_CACHE = False


def _cache_key(params: dict[str, Any]) -> str:
    raw = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _load_cache(key: str, bench_type: str) -> list[dict] | None:
    path = BENCHMARK_CACHE_DIR / f"{bench_type}_{key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _save_cache(key: str, bench_type: str, results: list[Any]) -> None:
    BENCHMARK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = BENCHMARK_CACHE_DIR / f"{bench_type}_{key}.json"
    data = [asdict(r) if hasattr(r, "__dataclass_fields__") else r for r in results]
    path.write_text(json.dumps(data, indent=2, default=str))

ROBUSTNESS_BATTERY: list[dict[str, str]] = [
    # Factual / knowledge
    {"prompt": "The capital of France is", "category": "factual", "subdomain": "geography"},
    {"prompt": "The boiling point of water is", "category": "factual", "subdomain": "science"},
    {
        "prompt": "Albert Einstein developed the theory of",
        "category": "factual",
        "subdomain": "physics",
    },
    {"prompt": "The chemical symbol for gold is", "category": "factual", "subdomain": "chemistry"},
    {
        "prompt": "The largest planet in our solar system is",
        "category": "factual",
        "subdomain": "astronomy",
    },
    # Reasoning / common sense
    {
        "prompt": "If all humans are mortal and Socrates is human, then",
        "category": "reasoning",
        "subdomain": "logic",
    },
    {"prompt": "A ball thrown in the air will", "category": "reasoning", "subdomain": "physics"},
    {
        "prompt": "If it rains, the ground gets wet. The ground is wet, therefore",
        "category": "reasoning",
        "subdomain": "logic",
    },
    {
        "prompt": "A triangle has three sides. A square has",
        "category": "reasoning",
        "subdomain": "geometry",
    },
    # Math / arithmetic
    {"prompt": "2 + 2 =", "category": "math", "subdomain": "arithmetic"},
    {"prompt": "The square root of 144 is", "category": "math", "subdomain": "algebra"},
    {"prompt": "10 factorial is", "category": "math", "subdomain": "combinatorics"},
    {
        "prompt": "If x = 5 and y = 3, then x * y + 2 =",
        "category": "math",
        "subdomain": "arithmetic",
    },
    # Code
    {"prompt": "def hello_world():\n    print(", "category": "code", "subdomain": "python"},
    {"prompt": "for i in range(10):\n    print(", "category": "code", "subdomain": "python"},
    # Creative / storytelling
    {"prompt": "Once upon a time", "category": "creative", "subdomain": "story"},
    {"prompt": "In a galaxy far far away", "category": "creative", "subdomain": "story"},
    {
        "prompt": "The old man walked to the edge of the cliff and",
        "category": "creative",
        "subdomain": "narrative",
    },
    # Instruction following
    {
        "prompt": "List three things you need to",
        "category": "instruction",
        "subdomain": "procedural",
    },
    {
        "prompt": "Explain the process of photosynthesis in",
        "category": "instruction",
        "subdomain": "explanation",
    },
    # Multilingual
    {"prompt": "Hola, ¿cómo estás?", "category": "multilingual", "subdomain": "spanish"},
    {"prompt": "Bonjour, comment allez-vous?", "category": "multilingual", "subdomain": "french"},
    # Domain specific
    {
        "prompt": "In quantum mechanics, the uncertainty principle states that",
        "category": "domain",
        "subdomain": "physics",
    },
    {
        "prompt": "The law of supply and demand states that",
        "category": "domain",
        "subdomain": "economics",
    },
    {"prompt": "The capital of Brazil is", "category": "factual", "subdomain": "geography"},
    {"prompt": "Python is a", "category": "domain", "subdomain": "programming"},
]


@dataclass
class BenchmarkResult:
    model: str
    strategy: str = "dynamic"
    tokens_generated: int = 0
    total_time_s: float = 0.0
    ttft_s: float = 0.0
    tokens_per_sec: float = 0.0
    decoding_tps: float = 0.0
    generation_tps: float = 0.0
    memory_mb: float = 0.0
    prompt_tokens: int = 0
    ouro_ppl: float = 0.0
    hrm_ppl: float = 0.0
    fused_ppl: float = 0.0
    avg_kl_oh: float = 0.0
    avg_kl_ho: float = 0.0
    avg_jsd: float = 0.0
    fusion_win_rate: float = 0.0
    avg_fusion_gain: float = 0.0
    oracle_rate: float = 0.0
    fused_entropy: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


def maybe_get_memory_mb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e6
    except Exception:
        pass
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return 0.0


def run_benchmark(
    text: str = "The quick brown fox jumps over the lazy dog.",
    max_new_tokens: int = 50,
    temperature: float = 0.0,
    top_k: int = 30,
    threshold: float = 0.01,
    ouro_weight: float = 0.5,
    local: bool = True,
    repetition_penalty: float = 1.0,
    condition: str = "direct",
    base_dir: str = "",
    configs: list[dict[str, Any]] | None = None,
    cache: bool = False,
    device: str = "auto",
) -> list[BenchmarkResult]:
    if configs is None:
        configs = [
            {"model": "ouro", "strategy": "average"},
            {"model": "hrm", "strategy": "average"},
            {"model": "fused", "strategy": "average"},
            {"model": "fused", "strategy": "product"},
            {"model": "fused", "strategy": "min-entropy"},
            {"model": "fused", "strategy": "cascade"},
            {"model": "fused", "strategy": "dynamic"},
            {"model": "fused", "strategy": "adaptive"},
            {"model": "fused", "strategy": "confidence"},
            {"model": "fused", "strategy": "hybrid"},
        ]

    if cache:
        ck = _cache_key({
            "text": text, "max_new_tokens": max_new_tokens,
            "temperature": temperature, "top_k": top_k, "threshold": threshold,
            "ouro_weight": ouro_weight, "configs": configs,
        })
        cached = _load_cache(ck, "speed")
        if cached is not None:
            print(f"  (loaded speed cache {ck})", file=sys.stderr)
            return [BenchmarkResult(**d) for d in cached]

    import torch
    from tokenizers import Tokenizer
    from transformers import AutoConfig, AutoModelForCausalLM

    from llm_fusion.fusion import Fuser, compute_kl, softmax_top_k
    from llm_fusion.generate import format_hrm_prompt
    from llm_fusion.loader import patch_ouro_model
    from llm_fusion.metrics import fusion_gain as _calc_gain
    from llm_fusion.metrics import parent_prob_for_token
    from llm_fusion.token_matcher import TokenMatcher

    bd = Path(base_dir) if base_dir else Path(__file__).resolve().parent.parent.parent
    ouro_tok_path = bd / "Ouro-1.4B/tokenizer.json"
    hrm_tok_path = bd / "HRM-Text-1B/tokenizer.json"
    matcher = TokenMatcher(str(ouro_tok_path), str(hrm_tok_path))
    ouro_tok = Tokenizer.from_file(str(ouro_tok_path))
    hrm_tok = Tokenizer.from_file(str(hrm_tok_path))

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cpu" else torch.float16

    needs_ouro = any(c["model"] in ("fused", "ouro") for c in configs)
    needs_hrm = any(c["model"] in ("fused", "hrm") for c in configs)

    ouro_model = None
    hrm_model = None

    if needs_ouro:
        ouro_model_path = str(bd / "Ouro-1.4B")
        ouro_config = AutoConfig.from_pretrained(ouro_model_path, trust_remote_code=True)
        patch_ouro_model(ouro_config)
        ouro_model = AutoModelForCausalLM.from_pretrained(
            ouro_model_path,
            config=ouro_config,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )

    if needs_hrm:
        hrm_model_path = str(bd / "HRM-Text-1B")
        hrm_model = AutoModelForCausalLM.from_pretrained(
            hrm_model_path,
            torch_dtype=dtype,
            device_map=device,
        )

    results: list[BenchmarkResult] = []

    for cfg in configs:
        model = cfg["model"]
        strategy = cfg.get("strategy", "dynamic")
        ouro_weight = cfg.get("ouro_weight", ouro_weight)

        r = BenchmarkResult(model=model, strategy=strategy)

        fuser = Fuser(matcher, ouro_tok, hrm_tok, ouro_weight, top_k, threshold, strategy)

        if model in ("ouro", "fused"):
            ouro_prompt_ids = ouro_tok.encode(text).ids
            r.prompt_tokens = len(ouro_prompt_ids)
        if model in ("hrm", "fused"):
            prompt = format_hrm_prompt(text, condition)
            hrm_ids = hrm_tok.encode(prompt).ids
            if model == "hrm":
                r.prompt_tokens = len(hrm_ids)

        generated_text = ""
        ouro_gen_ids: set[int] = set()
        hrm_gen_ids: set[int] = set()
        ouro_ids = list(ouro_prompt_ids) if model in ("fused", "ouro") else []
        hrm_ids_list = list(hrm_ids) if model in ("fused", "hrm") else []
        ouro_cache = None
        hrm_cache = None
        total_kl_oh = 0.0
        total_kl_ho = 0.0
        total_jsd = 0.0
        total_gain = 0.0
        fusion_wins = 0
        oracle_matches = 0
        total_entropy = 0.0
        n_kl_steps = 0
        ttft = 0.0
        t0 = time.time()

        for step in range(max_new_tokens):
            if model in ("fused", "ouro"):
                with torch.no_grad():
                    ouro_kwargs: dict[str, Any] = {}
                    if step > 0 and ouro_cache is not None:
                        ouro_kwargs["past_key_values"] = ouro_cache
                        ouro_kwargs["use_cache"] = True
                    ouro_out = ouro_model(
                        input_ids=torch.tensor([ouro_ids], device=device),
                        **ouro_kwargs,
                    )
                ouro_logits = ouro_out.logits[0, -1, :].tolist()
                if step == 0:
                    ouro_cache = ouro_out.past_key_values

            if model in ("fused", "hrm"):
                hrm_tti = torch.ones(
                    len(hrm_ids_list), dtype=torch.long, device=device
                ).unsqueeze(0)
                with torch.no_grad():
                    hrm_kwargs: dict[str, Any] = {}
                    if step > 0 and hrm_cache is not None:
                        hrm_kwargs["past_key_values"] = hrm_cache
                        hrm_kwargs["use_cache"] = True
                    hrm_out = hrm_model(
                        input_ids=torch.tensor([hrm_ids_list], device=device),
                        token_type_ids=hrm_tti,
                        **hrm_kwargs,
                    )
                hrm_logits = hrm_out.logits[0, -1, :].tolist()
                if step == 0:
                    hrm_cache = hrm_out.past_key_values

            if model == "fused":
                ouro_top_ids, ouro_probs = softmax_top_k(ouro_logits, top_k)
                hrm_top_ids, hrm_probs = softmax_top_k(hrm_logits, top_k)
                ouro_dist = dict(zip(ouro_top_ids, ouro_probs))
                hrm_dist = dict(zip(hrm_top_ids, hrm_probs))
                total_kl_oh += compute_kl(ouro_dist, hrm_dist)
                total_kl_ho += compute_kl(hrm_dist, ouro_dist)
                all_ids = set(ouro_dist) | set(hrm_dist)
                m_dist = {tid: 0.5 * (ouro_dist.get(tid, 0.0) + hrm_dist.get(tid, 0.0)) for tid in all_ids}
                total_jsd += 0.5 * compute_kl(ouro_dist, m_dist) + 0.5 * compute_kl(hrm_dist, m_dist)
                total_entropy += -sum(p * math.log(max(p, 1e-10)) for p in m_dist.values())
                n_kl_steps += 1
                fuser.current_step = step
                hrm_tid, ouro_tid, token_str, prob = fuser.sample_token_pair(ouro_logits, hrm_logits, temperature)
                ouro_p = parent_prob_for_token(ouro_logits, hrm_tid, top_k)
                hrm_p = parent_prob_for_token(hrm_logits, hrm_tid, top_k)
                gain = _calc_gain(prob, ouro_p, hrm_p)
                total_gain += gain
                if prob > max(ouro_p, hrm_p):
                    fusion_wins += 1
                if ouro_p >= hrm_p:
                    oracle_matches += 1
                hrm_ids_list = [hrm_tid]
                ouro_ids = [ouro_tid]
                hrm_gen_ids.add(hrm_tid)
            elif model == "ouro":
                from llm_fusion.generate import sample_from_logits

                tid, token_str, prob = sample_from_logits(ouro_logits, ouro_tok, top_k, temperature)
                ouro_ids = [tid]
                ouro_gen_ids.add(tid)
            elif model == "hrm":
                from llm_fusion.generate import sample_from_logits

                tid, token_str, prob = sample_from_logits(hrm_logits, hrm_tok, top_k, temperature)
                hrm_ids_list = [tid]
                hrm_gen_ids.add(tid)

            if step == 0:
                ttft = time.time() - t0

            if token_str:
                generated_text += token_str

        total = time.time() - t0
        r.tokens_generated = step + 1
        r.total_time_s = total
        r.ttft_s = ttft
        r.tokens_per_sec = (step + 1) / max(total, 1e-10)
        decode_time = max(total - ttft, 1e-10)
        r.decoding_tps = (step + 1) / decode_time
        r.generation_tps = (r.prompt_tokens + step + 1) / max(total, 1e-10)
        r.memory_mb = maybe_get_memory_mb()
        if model in ("ouro", "fused"):
            r.ouro_ppl = _quick_ppl(text, ouro_model, ouro_tok, device)
        if model in ("hrm", "fused"):
            hrm_formatted = format_hrm_prompt(text, "direct")
            r.hrm_ppl = _quick_ppl(hrm_formatted, hrm_model, hrm_tok, device)
        if model == "fused":
            r.fused_ppl = (r.ouro_ppl + r.hrm_ppl) / 2
        if model == "fused" and n_kl_steps > 0:
            r.avg_kl_oh = total_kl_oh / n_kl_steps
            r.avg_kl_ho = total_kl_ho / n_kl_steps
            r.avg_jsd = total_jsd / n_kl_steps
            r.avg_fusion_gain = total_gain / n_kl_steps
            r.fusion_win_rate = fusion_wins / n_kl_steps
            r.oracle_rate = oracle_matches / n_kl_steps
            r.fused_entropy = total_entropy / n_kl_steps
        results.append(r)

        label = f"{model}/{strategy}"
        print(
            f"  {label:30s}  decode={r.decoding_tps:7.1f} tok/s  "
            f"gen={r.generation_tps:7.1f} tok/s  "
            f"TTFT={r.ttft_s * 1000:.0f}ms  mem={r.memory_mb:.0f}MB",
            file=sys.stderr,
        )

    if cache:
        _save_cache(ck, "speed", results)

    return results


def format_table(results: list[BenchmarkResult]) -> str:
    lines = []
    lines.append(
        f"{'Config':30s}  {'Decode':>8s}  {'Gen':>8s}  {'FusedPPL':>9s}  {'KL(o>h)':>8s}  {'JSD':>6s}  {'WinRate':>8s}  {'Gain':>8s}  {'Oracle':>8s}  {'Entropy':>8s}"
    )
    lines.append("-" * 115)
    for r in results:
        label = f"{r.model}/{r.strategy}"
        lines.append(
            f"{label:30s}  {r.decoding_tps:7.1f}  {r.generation_tps:7.1f}  "
            f"{r.fused_ppl:9.1f}  {r.avg_kl_oh:8.3f}  {r.avg_jsd:6.3f}  "
            f"{r.fusion_win_rate:7.1%}  {r.avg_fusion_gain:+7.3f}  {r.oracle_rate:7.1%}  {r.fused_entropy:8.1f}"
        )
    return "\n".join(lines)


@dataclass
class RobustnessResult:
    prompt: str = ""
    category: str = ""
    subdomain: str = ""
    ouro_ppl: float = 0.0
    hrm_ppl: float = 0.0
    fused_ppl: float = 0.0
    avg_fusion_gain: float = 0.0
    fusion_win_rate: float = 0.0
    avg_kl_oh: float = 0.0
    avg_kl_ho: float = 0.0
    ouro_entropy: float = 0.0
    hrm_entropy: float = 0.0
    generated_len: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


def run_robustness_benchmark(
    max_new_tokens: int = 50,
    temperature: float = 0.0,
    top_k: int = 30,
    threshold: float = 0.01,
    ouro_weight: float = 0.5,
    _local: bool = True,
    base_dir: str = "",
    battery: list[dict[str, str]] | None = None,
    cache: bool = False,
    device: str = "auto",
) -> list[RobustnessResult]:
    if battery is None:
        battery = ROBUSTNESS_BATTERY

    if cache:
        ck = _cache_key({
            "max_new_tokens": max_new_tokens, "temperature": temperature,
            "top_k": top_k, "threshold": threshold, "ouro_weight": ouro_weight,
            "battery": [e["prompt"] for e in battery],
        })
        cached = _load_cache(ck, "robustness")
        if cached is not None:
            print(f"  (loaded robustness cache {ck})", file=sys.stderr)
            return [RobustnessResult(**d) for d in cached]

    import torch
    from tokenizers import Tokenizer
    from transformers import AutoConfig, AutoModelForCausalLM

    from llm_fusion.fusion import Fuser, compute_kl, softmax_top_k
    from llm_fusion.generate import format_hrm_prompt
    from llm_fusion.loader import patch_ouro_model
    from llm_fusion.metrics import fusion_gain as _calc_gain
    from llm_fusion.metrics import parent_prob_for_token
    from llm_fusion.token_matcher import TokenMatcher

    bd = Path(base_dir) if base_dir else Path(__file__).resolve().parent.parent.parent
    ouro_tok_path = bd / "Ouro-1.4B/tokenizer.json"
    hrm_tok_path = bd / "HRM-Text-1B/tokenizer.json"
    matcher = TokenMatcher(str(ouro_tok_path), str(hrm_tok_path))
    ouro_tok = Tokenizer.from_file(str(ouro_tok_path))
    hrm_tok = Tokenizer.from_file(str(hrm_tok_path))

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cpu" else torch.float16

    ouro_model_path = str(bd / "Ouro-1.4B")
    ouro_config = AutoConfig.from_pretrained(ouro_model_path, trust_remote_code=True)
    patch_ouro_model(ouro_config)
    ouro_model = AutoModelForCausalLM.from_pretrained(
        ouro_model_path,
        config=ouro_config,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )

    hrm_model_path = str(bd / "HRM-Text-1B")
    hrm_model = AutoModelForCausalLM.from_pretrained(
        hrm_model_path,
        torch_dtype=dtype,
        device_map=device,
    )

    fuser = Fuser(matcher, ouro_tok, hrm_tok, ouro_weight, top_k, threshold, "average")

    results: list[RobustnessResult] = []

    for entry in battery:
        prompt = entry["prompt"]
        cat = entry["category"]
        sub = entry.get("subdomain", "")

        hrm_prompt = format_hrm_prompt(prompt, "direct")
        hrm_ids_list = hrm_tok.encode(hrm_prompt).ids
        ouro_prompt_ids = ouro_tok.encode(prompt).ids or [0]

        if len(hrm_ids_list) < 2:
            continue

        total_kl_oh = 0.0
        total_kl_ho = 0.0
        total_gain = 0.0
        fusion_wins = 0
        n_steps = 0
        generated_text = ""

        for step in range(min(max_new_tokens, 30)):
            ouro_prefix_ids = (
                ouro_prompt_ids + ouro_tok.encode(generated_text).ids
                if generated_text
                else ouro_prompt_ids
            )
            with torch.no_grad():
                ouro_out = ouro_model(
                    input_ids=torch.tensor([ouro_prefix_ids], device=device),
                )
            ouro_logits = ouro_out.logits[0, -1, :].tolist()

            hrm_input_ids = hrm_ids_list
            with torch.no_grad():
                hrm_out = hrm_model(
                    input_ids=torch.tensor([hrm_input_ids], device=device),
                    token_type_ids=torch.ones(
                        len(hrm_input_ids), dtype=torch.long, device=device
                    ).unsqueeze(0),
                )
            hrm_logits = hrm_out.logits[0, -1, :].tolist()

            ouro_ids_k, ouro_probs = softmax_top_k(ouro_logits, top_k)
            hrm_ids_k, hrm_probs = softmax_top_k(hrm_logits, top_k)

            ouro_dist = dict(zip(ouro_ids_k, ouro_probs))
            hrm_dist = dict(zip(hrm_ids_k, hrm_probs))
            total_kl_oh += compute_kl(ouro_dist, hrm_dist)
            total_kl_ho += compute_kl(hrm_dist, ouro_dist)

            tid, token_str, prob = fuser.sample_token(ouro_logits, hrm_logits, temperature)
            ouro_p = parent_prob_for_token(ouro_logits, tid, top_k)
            hrm_p = parent_prob_for_token(hrm_logits, tid, top_k)
            total_gain += _calc_gain(prob, ouro_p, hrm_p)
            if prob > max(ouro_p, hrm_p):
                fusion_wins += 1

            hrm_ids_list.append(tid)
            if token_str:
                generated_text += token_str
            n_steps += 1

            if tid in (11, 0):
                break

        ouro_ppl = _quick_ppl(prompt, ouro_model, ouro_tok, device)
        hrm_ppl = _quick_ppl(format_hrm_prompt(prompt, "direct"), hrm_model, hrm_tok, device)

        results.append(
            RobustnessResult(
                prompt=prompt[:60],
                category=cat,
                subdomain=sub,
                ouro_ppl=ouro_ppl,
                hrm_ppl=hrm_ppl,
                fused_ppl=(ouro_ppl + hrm_ppl) / 2,
                avg_fusion_gain=total_gain / max(n_steps, 1),
                fusion_win_rate=fusion_wins / max(n_steps, 1),
                avg_kl_oh=total_kl_oh / max(n_steps, 1),
                avg_kl_ho=total_kl_ho / max(n_steps, 1),
                ouro_entropy=-sum(p * math.log(max(p, 1e-10)) for p in ouro_probs),
                hrm_entropy=-sum(p * math.log(max(p, 1e-10)) for p in hrm_probs),
                generated_len=n_steps,
            )
        )

    if cache:
        _save_cache(ck, "robustness", results)

    return results


def _quick_ppl(text: str, model: Any, tok: Any, device: str) -> float:
    """Compute perplexity quickly — single forward pass over the whole sequence."""
    import torch

    ids = tok.encode(text).ids
    if len(ids) < 2:
        return float("inf")
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids], device=device))
    logits = out.logits[0, :-1, :]
    targets = torch.tensor(ids[1:], device=device)
    ce = torch.nn.functional.cross_entropy(logits, targets)
    return float(math.exp(ce))


def format_robustness_table(
    results: list[RobustnessResult],
    group_by: str = "category",
) -> str:
    lines = []
    if not results:
        return "  (no results)"
    groups: dict[str, list[RobustnessResult]] = {}
    for r in results:
        key = getattr(r, group_by, "other")
        groups.setdefault(key, []).append(r)

    for group_name in sorted(groups):
        items = groups[group_name]
        n = len(items)

        avg_ouro_ppl = sum(r.ouro_ppl for r in items) / n
        avg_hrm_ppl = sum(r.hrm_ppl for r in items) / n
        avg_fused_ppl = sum(r.fused_ppl for r in items) / n
        avg_gain = sum(r.avg_fusion_gain for r in items) / n
        avg_win = sum(r.fusion_win_rate for r in items) / n
        avg_kl = sum(r.avg_kl_oh for r in items) / n

        lines.append(f"\n  [{group_name}]  ({n} prompts)")
        lines.append(
            f"    {'Metric':25s}  {'Ouro':>8s}  {'HRM':>8s}  {'Fused':>8s}  {'Fusion':>8s}"
        )
        lines.append(f"    {'-' * 25}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 8}")
        lines.append(
            f"    {'Perplexity':25s}  {avg_ouro_ppl:8.1f}  {avg_hrm_ppl:8.1f}  {avg_fused_ppl:8.1f}  {'':>8s}"
        )
        lines.append(f"    {'Fusion Gain':25s}  {'':>8s}  {'':>8s}  {'':>8s}  {avg_gain:+8.3f}")
        lines.append(f"    {'Fusion Win Rate':25s}  {'':>8s}  {'':>8s}  {'':>8s}  {avg_win:7.1%}")
        lines.append(f"    {'Avg KL(O||H)':25s}  {'':>8s}  {'':>8s}  {'':>8s}  {avg_kl:8.2f}")

    lines.append(f"\n  {'TOTAL':25s}  ({len(results)} prompts)")
    if results:
        all_gains = [r.avg_fusion_gain for r in results]
        all_wins = [r.fusion_win_rate for r in results]
        all_kl = [r.avg_kl_oh for r in results]
        mean_gain = sum(all_gains) / len(all_gains)
        lines.append(f"    Mean fusion gain:  {mean_gain:+.4f}")
        lines.append(f"    Mean fusion win:   {sum(all_wins) / len(all_wins):.1%}")
        lines.append(f"    Mean KL(O||H):     {sum(all_kl) / len(all_kl):.2f}")
        lines.append(
            f"    Fusion outperforms best parent on avg: {'YES' if mean_gain > 0 else 'NO'}"
        )

    return "\n".join(lines)


def run_graph_benchmark(
    text: str = "The quick brown fox jumps over the lazy dog.",
    temperature: float = 0.0,
    cache: bool = False,
    device: str = "auto",
) -> None:
    import matplotlib.pyplot as plt

    token_counts = [10, 30, 50]
    strategies = ["average", "product", "min-entropy", "cascade", "dynamic"]
    metrics = ["fusion_win_rate", "avg_fusion_gain", "avg_kl_oh", "avg_jsd", "fused_entropy"]
    metric_labels = {
        "fusion_win_rate": "Win Rate",
        "avg_fusion_gain": "Fusion Gain",
        "avg_kl_oh": "KL(O||H)",
        "avg_jsd": "JSD",
        "fused_entropy": "Entropy",
    }

    all_results: dict[int, list[BenchmarkResult]] = {}
    for n in token_counts:
        print(f"\nRunning benchmark for n={n}...", file=sys.stderr)
        configs = [{"model": "fused", "strategy": s} for s in strategies]
        results = run_benchmark(
            text=text,
            max_new_tokens=n,
            temperature=temperature,
            cache=cache,
            device=device,
            configs=configs,
        )
        all_results[n] = results

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Fusion Metrics vs Completion Tokens", fontsize=14)

    for ax, metric in zip(axes.flat, metrics):
        for strategy in strategies:
            xs = []
            ys = []
            for n in token_counts:
                for r in all_results[n]:
                    if r.strategy == strategy:
                        xs.append(n)
                        ys.append(getattr(r, metric, 0.0))
            ax.plot(xs, ys, marker="o", label=strategy, linewidth=2)
        ax.set_xlabel("Completion Tokens (n)")
        ax.set_ylabel(metric_labels.get(metric, metric))
        ax.set_title(metric_labels.get(metric, metric))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    if len(metrics) < len(axes.flat):
        for ax in axes.flat[len(metrics):]:
            ax.set_visible(False)

    plt.tight_layout()
    out_path = "benchmark_metrics_vs_n.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nGraph saved to {out_path}", file=sys.stderr)
    plt.close()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="LLM Fusion benchmarks")
    parser.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog.")
    parser.add_argument("-n", "--max-new-tokens", type=int, default=50)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument(
        "--robustness",
        action="store_true",
        help="Run diverse robustness battery instead of speed benchmark",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--debug", action="store_true", help="Debug output")
    parser.add_argument(
        "--benchmark-cache",
        action="store_true",
        help="Cache results to .benchmark_cache/ and load from cache if available",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["cpu", "cuda", "auto"],
        help="Device to run on (default: auto)",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="Run benchmarks at multiple token counts and plot metrics vs n",
    )
    args = parser.parse_args()

    level = logging.DEBUG if args.debug else (logging.INFO if args.verbose else logging.WARNING)
    logging.basicConfig(
        level=level,
        format="%(levelname)-5s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if args.robustness:
        print("Running robustness benchmark on diverse battery...", file=sys.stderr)
        print(f"  {len(ROBUSTNESS_BATTERY)} prompts across multiple categories", file=sys.stderr)
        print("-" * 60, file=sys.stderr)
        results = run_robustness_benchmark(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temp,
            _local=True,
            cache=args.benchmark_cache,
            device=args.device,
        )
        print("\n" + format_robustness_table(results))
    else:
        results = run_benchmark(
            text=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temp,
            cache=args.benchmark_cache,
            device=args.device,
        )
        print("\n" + format_table(results))

    if args.graph:
        run_graph_benchmark(
            text=args.prompt,
            temperature=args.temp,
            cache=args.benchmark_cache,
            device=args.device,
        )


if __name__ == "__main__":
    raise SystemExit(main())
