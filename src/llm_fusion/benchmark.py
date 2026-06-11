"""Benchmark generation speed and memory for model configurations."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkResult:
    model: str
    strategy: str = "average"
    tokens_generated: int = 0
    total_time_s: float = 0.0
    ttft_s: float = 0.0
    tokens_per_sec: float = 0.0
    memory_mb: float = 0.0
    prompt_tokens: int = 0
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
        ]

    import torch
    from tokenizers import Tokenizer
    from transformers import AutoConfig, AutoModelForCausalLM

    from llm_fusion.fusion import Fuser
    from llm_fusion.generate import format_hrm_prompt, patch_ouro_model
    from llm_fusion.token_matcher import TokenMatcher

    bd = Path(base_dir) if base_dir else Path(__file__).resolve().parent.parent.parent
    ouro_tok_path = bd / "Ouro-1.4B/tokenizer.json"
    hrm_tok_path = bd / "HRM-Text-1B/tokenizer.json"
    matcher = TokenMatcher(str(ouro_tok_path), str(hrm_tok_path))
    ouro_tok = Tokenizer.from_file(str(ouro_tok_path))
    hrm_tok = Tokenizer.from_file(str(hrm_tok_path))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    needs_ouro = any(c["model"] in ("fused", "ouro") for c in configs)
    needs_hrm = any(c["model"] in ("fused", "hrm") for c in configs)

    ouro_model = None
    hrm_model = None

    if needs_ouro:
        ouro_model_path = str(bd / "Ouro-1.4B")
        ouro_config = AutoConfig.from_pretrained(ouro_model_path, trust_remote_code=True)
        patch_ouro_model(ouro_config)
        ouro_model = AutoModelForCausalLM.from_pretrained(
            ouro_model_path, config=ouro_config, torch_dtype=dtype,
            device_map=device, trust_remote_code=True,
        )

    if needs_hrm:
        hrm_model_path = str(bd / "HRM-Text-1B")
        hrm_model = AutoModelForCausalLM.from_pretrained(
            hrm_model_path, torch_dtype=dtype, device_map=device,
        )

    results: list[BenchmarkResult] = []

    for cfg in configs:
        model = cfg["model"]
        strategy = cfg.get("strategy", "average")
        ouro_weight = cfg.get("ouro_weight", ouro_weight)

        r = BenchmarkResult(model=model, strategy=strategy)

        fuser = Fuser(matcher, ouro_tok, hrm_tok, ouro_weight, top_k, threshold, strategy)

        if model in ("ouro", "fused"):
            ouro_prompt_ids = ouro_tok.encode(text).ids
            r.prompt_tokens = len(ouro_prompt_ids)
        else:
            prompt = format_hrm_prompt(text, condition)
            hrm_ids = hrm_tok.encode(prompt).ids
            r.prompt_tokens = len(hrm_ids)

        generated_text = ""
        ouro_gen_ids: set[int] = set()
        hrm_gen_ids: set[int] = set()
        ouro_ids = list(ouro_prompt_ids) if model in ("fused", "ouro") else []
        hrm_ids_list = list(hrm_ids) if model in ("fused", "hrm") else []
        ttft = 0.0
        t0 = time.time()

        for step in range(max_new_tokens):
            if model in ("fused", "ouro"):
                if model == "fused":
                    ouro_ids = ouro_prompt_ids + ouro_tok.encode(generated_text).ids
                with torch.no_grad():
                    ouro_out = ouro_model(
                        input_ids=torch.tensor([ouro_ids], device=device),
                    )
                ouro_logits = ouro_out.logits[0, -1, :].tolist()

            if model in ("fused", "hrm"):
                if model == "fused":
                    hrm_tti = torch.ones(len(hrm_ids_list), dtype=torch.long, device=device).unsqueeze(0)
                    with torch.no_grad():
                        hrm_out = hrm_model(
                            input_ids=torch.tensor([hrm_ids_list], device=device),
                            token_type_ids=hrm_tti,
                        )
                    hrm_logits = hrm_out.logits[0, -1, :].tolist()

            if model == "fused":
                fuser.current_step = step
                tid, token_str, prob = fuser.sample_token(ouro_logits, hrm_logits, temperature)
                hrm_ids_list.append(tid)
                hrm_gen_ids.add(tid)
            elif model == "ouro":
                from llm_fusion.generate import sample_from_logits
                tid, token_str, prob = sample_from_logits(ouro_logits, ouro_tok, top_k, temperature)
                ouro_ids.append(tid)
                ouro_gen_ids.add(tid)
            elif model == "hrm":
                from llm_fusion.generate import sample_from_logits
                tid, token_str, prob = sample_from_logits(hrm_logits, hrm_tok, top_k, temperature)
                hrm_ids_list.append(tid)
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
        r.memory_mb = maybe_get_memory_mb()
        results.append(r)

        label = f"{model}/{strategy}"
        print(f"  {label:30s}  {r.tokens_per_sec:7.1f} tok/s  "
              f"TTFT={r.ttft_s*1000:.0f}ms  mem={r.memory_mb:.0f}MB",
              file=sys.stderr)

    return results


def format_table(results: list[BenchmarkResult]) -> str:
    lines = []
    lines.append(f"{'Config':30s}  {'tok/s':>7s}  {'TTFT':>6s}  {'Tokens':>6s}  {'Mem':>6s}")
    lines.append("-" * 65)
    for r in results:
        label = f"{r.model}/{r.strategy}"
        lines.append(f"{label:30s}  {r.tokens_per_sec:7.1f}  {r.ttft_s*1000:4.0f}ms  "
                     f"{r.tokens_generated:5d}   {r.memory_mb:5.0f}MB")
    return "\n".join(lines)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark generation speed")
    parser.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog.")
    parser.add_argument("-n", "--max-new-tokens", type=int, default=50)
    parser.add_argument("--temp", type=float, default=0.0)
    args = parser.parse_args()

    results = run_benchmark(
        text=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temp,
    )
    print("\n" + format_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
