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
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"

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

def save_results(results: list[Any], tag: str = "speed") -> Path:
    """Append benchmark results to a timestamped JSON file in results/."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"benchmark_{tag}_{ts}.json"
    data = [asdict(r) if hasattr(r, "__dataclass_fields__") else r for r in results]
    out.write_text(json.dumps(data, indent=2, default=str))
    print(f"  Results saved to {out}", file=sys.stderr)
    return out


def load_results(path: str | Path) -> list[dict[str, Any]]:
    """Load benchmark results from a JSON file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Results file not found: {p}")
    return json.loads(p.read_text())


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
    {"prompt": "DNA stands for", "category": "factual", "subdomain": "biology"},
    {"prompt": "The painter of the Mona Lisa is", "category": "factual", "subdomain": "art"},
    # Reasoning / common sense
    {
        "prompt": "If all humans are mortal and Socrates is human, then",
        "category": "reasoning",
        "subdomain": "logic",
    },
    {"prompt": "A ball thrown in the air will", "category": "reasoning", "subdomain": "physics"},
    {"prompt": "A car needs fuel to", "category": "reasoning", "subdomain": "common_sense"},
    {"prompt": "The opposite of hot is", "category": "reasoning", "subdomain": "language"},
    # Math / arithmetic
    {"prompt": "2 + 2 =", "category": "math", "subdomain": "arithmetic"},
    {"prompt": "The square root of 144 is", "category": "math", "subdomain": "algebra"},
    {"prompt": "10 factorial is", "category": "math", "subdomain": "combinatorics"},
    {"prompt": "The derivative of x squared is", "category": "math", "subdomain": "calculus"},
    # Code
    {"prompt": "def hello_world():\n    print(", "category": "code", "subdomain": "python"},
    {"prompt": "SELECT * FROM users WHERE", "category": "code", "subdomain": "sql"},
    {"prompt": "#include <stdio.h>\nint main() {", "category": "code", "subdomain": "c"},
    # Creative / storytelling
    {"prompt": "Once upon a time", "category": "creative", "subdomain": "story"},
    {"prompt": "She opened the door and saw", "category": "creative", "subdomain": "narrative"},
    {"prompt": "In the year 2050, humans will", "category": "creative", "subdomain": "sci_fi"},
    # Instruction following
    {
        "prompt": "Explain the process of photosynthesis in",
        "category": "instruction",
        "subdomain": "explanation",
    },
    {"prompt": "To make a cup of coffee, you need to", "category": "instruction", "subdomain": "procedural"},
    {"prompt": "The key differences between Python and Java are", "category": "instruction", "subdomain": "technical"},
    # Multilingual
    {"prompt": "Hola, ¿cómo estás?", "category": "multilingual", "subdomain": "spanish"},
    {"prompt": "Bonjour, comment allez-vous?", "category": "multilingual", "subdomain": "french"},
    # Historical
    {"prompt": "World War II ended in", "category": "historical", "subdomain": "events"},
    {"prompt": "The first president of the United States was", "category": "historical", "subdomain": "politics"},
    # Medical / health
    {"prompt": "The human heart has", "category": "medical", "subdomain": "anatomy"},
    {"prompt": "Vaccines work by", "category": "medical", "subdomain": "immunology"},
    # Technical / scientific
    {"prompt": "TCP/IP is a", "category": "technical", "subdomain": "networking"},
    {"prompt": "Machine learning is a subset of", "category": "technical", "subdomain": "ai"},
    {"prompt": "The periodic table contains", "category": "technical", "subdomain": "chemistry"},
    # Conversational / social
    {"prompt": "How are you today?", "category": "conversational", "subdomain": "greeting"},
    {"prompt": "The best way to learn programming is to", "category": "conversational", "subdomain": "advice"},
    # Legal / formal
    {"prompt": "The First Amendment protects", "category": "legal", "subdomain": "constitutional"},
    {"prompt": "A contract requires", "category": "legal", "subdomain": "civil_law"},
    # Edge cases / adversarial
    {"prompt": "", "category": "edge", "subdomain": "empty"},
    {"prompt": "a", "category": "edge", "subdomain": "minimal"},
    {"prompt": "???", "category": "edge", "subdomain": "punctuation"},
    # Complex coding
    {
        "prompt": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return ",
        "category": "code",
        "subdomain": "python_recursive",
    },
    {
        "prompt": "class LinkedList:\n    def __init__(self):\n        self.head = None\n\n    def append(self, value):\n        new_node = ",
        "category": "code",
        "subdomain": "python_data_structure",
    },
    {
        "prompt": "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = ",
        "category": "code",
        "subdomain": "python_algorithm",
    },
    {
        "prompt": "CREATE TABLE users (\n    id INTEGER PRIMARY KEY,\n    name VARCHAR(100) NOT NULL,\n    email ",
        "category": "code",
        "subdomain": "sql_ddl",
    },
    {
        "prompt": "async function fetchUserData(userId) {\n    const response = await fetch(",
        "category": "code",
        "subdomain": "javascript_async",
    },
    {
        "prompt": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while ",
        "category": "code",
        "subdomain": "python_search",
    },
    # Complex math
    {
        "prompt": "The integral of x^2 dx is",
        "category": "math",
        "subdomain": "calculus_integration",
    },
    {
        "prompt": "The eigenvalues of the matrix [[2, 1], [1, 2]] are",
        "category": "math",
        "subdomain": "linear_algebra",
    },
    {
        "prompt": "Using the quadratic formula for x^2 - 5x + 6 = 0, the solutions are",
        "category": "math",
        "subdomain": "quadratic",
    },
    {
        "prompt": "The limit of sin(x)/x as x approaches 0 is",
        "category": "math",
        "subdomain": "limits",
    },
    {
        "prompt": "The determinant of the matrix [[1, 2], [3, 4]] is",
        "category": "math",
        "subdomain": "determinant",
    },
    # Complex reasoning
    {
        "prompt": "Write a Python function that checks if a string is a palindrome:\ndef is_palindrome(s):\n    ",
        "category": "code",
        "subdomain": "python_string",
    },
    {
        "prompt": "Explain the difference between a stack and a queue. A stack is",
        "category": "reasoning",
        "subdomain": "data_structures",
    },
    {
        "prompt": "What are the time complexities of bubble sort, merge sort, and quicksort? Bubble sort is",
        "category": "reasoning",
        "subdomain": "algorithm_analysis",
    },
    {
        "prompt": "Write a SQL query to find all users who signed up in the last 30 days:\nSELECT * FROM users WHERE ",
        "category": "code",
        "subdomain": "sql_query",
    },
    {
        "prompt": "The three main principles of object-oriented programming are encapsulation, inheritance, and",
        "category": "technical",
        "subdomain": "oop",
    },
    # Multi-step problems
    {
        "prompt": "Solve for x: 2x + 5 = 13. First, subtract 5 from both sides to get 2x = ",
        "category": "math",
        "subdomain": "algebra_steps",
    },
    {
        "prompt": "To deploy a Flask app to production, you would first ",
        "category": "technical",
        "subdomain": "deployment",
    },
    {
        "prompt": "The time complexity of building a heap from an unsorted array is O(",
        "category": "technical",
        "subdomain": "complexity",
    },
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
    ouro_tokens_used: int = 0
    hrm_tokens_used: int = 0
    prompt: str = ""
    completion: str = ""
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


@dataclass
class LoadedModels:
    matcher: Any
    ouro_tok: Any
    hrm_tok: Any
    ouro_model: Any = None
    hrm_model: Any = None
    device: str = "cpu"


def load_models(
    base_dir: str = "",
    device: str = "auto",
    local: bool = True,
) -> LoadedModels:
    import torch
    from tokenizers import Tokenizer
    from transformers import AutoConfig, AutoModelForCausalLM

    from llm_fusion.loader import patch_ouro_model
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

    print(f"Loading models on {device}...", file=sys.stderr)

    ouro_model_path = str(bd / "Ouro-1.4B") if local else "ByteDance/Ouro-1.4B"
    ouro_config = AutoConfig.from_pretrained(ouro_model_path, trust_remote_code=True)
    patch_ouro_model(ouro_config)
    ouro_model = AutoModelForCausalLM.from_pretrained(
        ouro_model_path,
        config=ouro_config,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )

    hrm_model_path = str(bd / "HRM-Text-1B") if local else "sapientinc/HRM-Text-1B"
    hrm_model = AutoModelForCausalLM.from_pretrained(
        hrm_model_path,
        torch_dtype=dtype,
        device_map=device,
        attn_implementation="sdpa",
    )

    print(f"Models loaded on {device}", file=sys.stderr)
    return LoadedModels(matcher=matcher, ouro_tok=ouro_tok, hrm_tok=hrm_tok,
                        ouro_model=ouro_model, hrm_model=hrm_model, device=device)


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
    loaded: LoadedModels | None = None,
    show_completions: bool = False,
) -> list[BenchmarkResult]:
    if configs is None:
        configs = [
            {"model": "ouro", "strategy": "average"},
            {"model": "hrm", "strategy": "average"},
            {"model": "fused", "strategy": "average"},
            {"model": "fused", "strategy": "product"},
            {"model": "fused", "strategy": "min-entropy"},
            {"model": "fused", "strategy": "min-perplexity"},
            {"model": "fused", "strategy": "cascade"},
            {"model": "fused", "strategy": "dynamic"},
            {"model": "fused", "strategy": "adaptive"},
            {"model": "fused", "strategy": "confidence"},
            {"model": "fused", "strategy": "hybrid"},
            {"model": "fused", "strategy": "slerp"},
            {"model": "fused", "strategy": "simple"},
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

    if loaded is not None:
        matcher = loaded.matcher
        ouro_tok = loaded.ouro_tok
        hrm_tok = loaded.hrm_tok
        ouro_model = loaded.ouro_model
        hrm_model = loaded.hrm_model
        device = loaded.device
    else:
        import torch
        from tokenizers import Tokenizer
        from transformers import AutoConfig, AutoModelForCausalLM

        from llm_fusion.loader import patch_ouro_model
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
                attn_implementation="sdpa",
            )

    import torch
    from llm_fusion.fusion import Fuser, compute_kl, softmax_top_k, softmax_top_k_torch
    from llm_fusion.generate import format_hrm_prompt, HRM_EOS_ID, OURO_EOS_ID, apply_repetition_penalty
    from llm_fusion.metrics import fusion_gain as _calc_gain
    from llm_fusion.metrics import parent_prob_for_token

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
        ouro_tokens_used = 0
        hrm_tokens_used = 0
        ttft = 0.0
        t0 = time.time()

        _single_token = torch.tensor([[0]], device=device, dtype=torch.long)
        _hrm_tti_1 = torch.ones(1, dtype=torch.long, device=device).unsqueeze(0)

        for step in range(max_new_tokens):
            if model in ("fused", "ouro"):
                with torch.no_grad():
                    ouro_kwargs: dict[str, Any] = {}
                    if step > 0 and ouro_cache is not None:
                        ouro_kwargs["past_key_values"] = ouro_cache
                        ouro_kwargs["use_cache"] = True
                        _single_token[0] = ouro_ids[0]
                        ouro_input = _single_token
                    else:
                        ouro_input = torch.tensor([ouro_ids], device=device)
                    ouro_out = ouro_model(
                        input_ids=ouro_input,
                        **ouro_kwargs,
                    )
                ouro_logits_t = ouro_out.logits[0, -1, :]
                if step == 0:
                    ouro_cache = ouro_out.past_key_values
                if repetition_penalty != 1.0:
                    ouro_logits_t = torch.tensor(
                        apply_repetition_penalty(ouro_logits_t.tolist(), ouro_gen_ids, repetition_penalty),
                        device=device,
                    )

            if model in ("fused", "hrm"):
                with torch.no_grad():
                    hrm_kwargs: dict[str, Any] = {}
                    if step > 0 and hrm_cache is not None:
                        hrm_kwargs["past_key_values"] = hrm_cache
                        hrm_kwargs["use_cache"] = True
                        _single_token[0] = hrm_ids_list[0]
                        hrm_out = hrm_model(
                            input_ids=_single_token,
                            token_type_ids=_hrm_tti_1,
                            **hrm_kwargs,
                        )
                    else:
                        hrm_tti = torch.ones(
                            len(hrm_ids_list), dtype=torch.long, device=device
                        ).unsqueeze(0)
                        hrm_out = hrm_model(
                            input_ids=torch.tensor([hrm_ids_list], device=device),
                            token_type_ids=hrm_tti,
                        )
                hrm_logits_t = hrm_out.logits[0, -1, :]
                if step == 0:
                    hrm_cache = hrm_out.past_key_values
                if repetition_penalty != 1.0:
                    hrm_logits_t = torch.tensor(
                        apply_repetition_penalty(hrm_logits_t.tolist(), hrm_gen_ids, repetition_penalty),
                        device=device,
                    )

            if model == "fused":
                ouro_topk_vals, ouro_topk_ids_t = torch.topk(ouro_logits_t, top_k)
                hrm_topk_vals, hrm_topk_ids_t = torch.topk(hrm_logits_t, top_k)
                ouro_probs_t = torch.softmax(ouro_topk_vals, dim=0)
                hrm_probs_t = torch.softmax(hrm_topk_vals, dim=0)
                ouro_top_ids = ouro_topk_ids_t.tolist()
                ouro_probs = ouro_probs_t.tolist()
                hrm_top_ids = hrm_topk_ids_t.tolist()
                hrm_probs = hrm_probs_t.tolist()
                ouro_dist = dict(zip(ouro_top_ids, ouro_probs))
                hrm_dist = dict(zip(hrm_top_ids, hrm_probs))

                _log_1e10 = math.log(1e-10)
                kl_oh = 0.0
                kl_ho = 0.0
                jsd = 0.0
                entropy = 0.0
                for hid, hp in zip(hrm_top_ids, hrm_probs):
                    if hp > 0:
                        op = ouro_dist.get(hid, 0.0)
                        kl_oh += hp * (math.log(hp) - math.log(max(op, 1e-10)))
                        m = 0.5 * (hp + op)
                        if m > 0:
                            ml = math.log(m)
                            jsd += 0.5 * hp * (math.log(hp) - ml)
                            entropy -= m * ml
                for oid, op in zip(ouro_top_ids, ouro_probs):
                    if op > 0:
                        hp = hrm_dist.get(oid, 0.0)
                        kl_ho += op * (math.log(op) - math.log(max(hp, 1e-10)))
                        m = 0.5 * (op + hp)
                        if m > 0:
                            ml = math.log(m)
                            jsd += 0.5 * op * (math.log(op) - ml)
                total_kl_oh += kl_oh
                total_kl_ho += kl_ho
                total_jsd += jsd
                total_entropy += entropy
                n_kl_steps += 1

                fuser.current_step = step
                ouro_logits_list = ouro_logits_t.tolist()
                hrm_logits_list = hrm_logits_t.tolist()
                hrm_tid, ouro_tid, token_str, prob = fuser.sample_token_pair(ouro_logits_list, hrm_logits_list, temperature)

                ouro_p = parent_prob_for_token(ouro_logits_list, hrm_tid, top_k)
                hrm_p = hrm_dist.get(hrm_tid, 0.0)

                gain = _calc_gain(prob, ouro_p, hrm_p)
                total_gain += gain
                if prob > max(ouro_p, hrm_p):
                    fusion_wins += 1
                if ouro_p >= hrm_p:
                    oracle_matches += 1
                hrm_ids_list = [hrm_tid]
                ouro_ids = [ouro_tid]
                hrm_gen_ids.add(hrm_tid)
                if strategy in ("min-entropy", "min-perplexity") and fuser.last_routed_model == "ouro":
                    ouro_tokens_used += 1
                elif strategy in ("min-entropy", "min-perplexity") and fuser.last_routed_model == "hrm":
                    hrm_tokens_used += 1
                elif strategy == "cascade":
                    if ouro_p >= hrm_p:
                        ouro_tokens_used += 1
                    else:
                        hrm_tokens_used += 1
                else:
                    ouro_tokens_used += 1
                    hrm_tokens_used += 1
            elif model == "ouro":
                from llm_fusion.generate import sample_from_logits

                tid, token_str, prob = sample_from_logits(ouro_logits_t.tolist(), ouro_tok, top_k, temperature)
                ouro_ids = [tid]
                ouro_gen_ids.add(tid)
            elif model == "hrm":
                from llm_fusion.generate import sample_from_logits

                tid, token_str, prob = sample_from_logits(hrm_logits_t.tolist(), hrm_tok, top_k, temperature)
                hrm_ids_list = [tid]
                hrm_gen_ids.add(tid)

            if step == 0:
                ttft = time.time() - t0

            check_tid = hrm_tid if model == "fused" else (tid if model in ("ouro", "hrm") else None)
            if check_tid is not None:
                eos = HRM_EOS_ID if model in ("fused", "hrm") else OURO_EOS_ID
                if check_tid == eos:
                    break

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
        r.prompt = text
        r.completion = generated_text
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
        r.ouro_tokens_used = ouro_tokens_used
        r.hrm_tokens_used = hrm_tokens_used
        results.append(r)

        label = f"{model}/{strategy}"
        print(
            f"  {label:30s}  decode={r.decoding_tps:7.1f} tok/s  "
            f"gen={r.generation_tps:7.1f} tok/s  "
            f"TTFT={r.ttft_s * 1000:.0f}ms  mem={r.memory_mb:.0f}MB",
            file=sys.stderr,
        )
        if show_completions and r.prompt:
            prompt_short = r.prompt.replace("\n", "\\n")
            completion_short = r.completion.replace("\n", "\\n") if r.completion else ""
            print(f"    prompt:     {prompt_short}", file=sys.stderr)
            print(f"    completion: {completion_short}", file=sys.stderr)

    if cache:
        _save_cache(ck, "speed", results)

    return results


def format_table(results: list[BenchmarkResult], show_completions: bool = False) -> str:
    lines = []
    lines.append(
        f"{'Config':30s}  {'Decode':>8s}  {'Gen':>8s}  {'FusedPPL':>9s}  {'KL(o>h)':>8s}  {'JSD':>6s}  {'WinRate':>8s}  {'Gain':>8s}  {'Oracle':>8s}  {'Entropy':>8s}  {'OuroTok':>8s}  {'HrmTok':>8s}"
    )
    lines.append("-" * 143)
    for r in results:
        label = f"{r.model}/{r.strategy}"
        lines.append(
            f"{label:30s}  {r.decoding_tps:7.1f}  {r.generation_tps:7.1f}  "
            f"{r.fused_ppl:9.1f}  {r.avg_kl_oh:8.3f}  {r.avg_jsd:6.3f}  "
            f"{r.fusion_win_rate:7.1%}  {r.avg_fusion_gain:+7.3f}  {r.oracle_rate:7.1%}  {r.fused_entropy:8.1f}  {r.ouro_tokens_used:8d}  {r.hrm_tokens_used:8d}"
        )
        if show_completions and r.prompt:
            prompt_short = r.prompt.replace("\n", "\\n")
            completion_short = r.completion.replace("\n", "\\n") if r.completion else ""
            lines.append(f"  prompt:     {prompt_short}")
            lines.append(f"  completion: {completion_short}")
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
        attn_implementation="sdpa",
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
    results_file: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    token_counts = [10, 30, 50]
    strategies = ["average", "product", "min-entropy", "min-perplexity", "cascade", "dynamic", "slerp", "simple"]
    metrics = ["fusion_win_rate", "avg_fusion_gain", "avg_kl_oh", "avg_jsd", "fused_entropy"]
    metric_labels = {
        "fusion_win_rate": "Win Rate",
        "avg_fusion_gain": "Fusion Gain",
        "avg_kl_oh": "KL(O||H)",
        "avg_jsd": "JSD",
        "fused_entropy": "Entropy",
    }

    all_results: dict[int, list[BenchmarkResult]] = {}

    if results_file:
        raw = load_results(results_file)
        for d in raw:
            n = d.get("tokens_generated", 0)
            all_results.setdefault(n, []).append(BenchmarkResult(**d))
        token_counts = sorted(all_results.keys())
        print(f"  Loaded {len(raw)} results from {results_file}", file=sys.stderr)
    else:
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
    parser.add_argument(
        "--single",
        action="store_true",
        help="Run on single prompt only (default: run all prompts and average)",
    )
    parser.add_argument(
        "--show-completions",
        action="store_true",
        help="Show prompt and completion text for each (prompt, strategy) pair",
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
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to results/ directory (always appends a timestamped JSON)",
    )
    parser.add_argument(
        "--results-file",
        default=None,
        help="Path to a results JSON file to ingest (for --graph or display)",
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
        tag = "robustness"
        print("\n" + format_robustness_table(results))
    else:
        loaded = load_models(device=args.device)
        if args.single:
            results = run_benchmark(
                text=args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temp,
                cache=args.benchmark_cache,
                device=args.device,
                loaded=loaded,
                show_completions=args.show_completions,
            )
            tag = "speed"
            print("\n" + format_table(results, show_completions=args.show_completions))
        else:
            prompts = [b["prompt"] for b in ROBUSTNESS_BATTERY]
            print(f"Running benchmark on {len(prompts)} prompts...", file=sys.stderr)
            all_results: dict[str, list[BenchmarkResult]] = {}
            for i, prompt in enumerate(prompts):
                print(f"  [{i+1}/{len(prompts)}] {prompt[:40]}...", file=sys.stderr)
                bench_results = run_benchmark(
                    text=prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temp,
                    cache=args.benchmark_cache,
                    device=args.device,
                    loaded=loaded,
                    show_completions=args.show_completions,
                )
                for r in bench_results:
                    key = f"{r.model}/{r.strategy}"
                    if key not in all_results:
                        all_results[key] = []
                    all_results[key].append(r)

            averaged: list[BenchmarkResult] = []
            for key in sorted(all_results):
                rs = all_results[key]
                n = len(rs)
                averaged.append(BenchmarkResult(
                    model=rs[0].model,
                    strategy=rs[0].strategy,
                    tokens_generated=round(sum(r.tokens_generated for r in rs) / n),
                    decoding_tps=sum(r.decoding_tps for r in rs) / n,
                generation_tps=sum(r.generation_tps for r in rs) / n,
                ttft_s=sum(r.ttft_s for r in rs) / n,
                memory_mb=max(r.memory_mb for r in rs),
                fused_ppl=sum(r.fused_ppl for r in rs) / n,
                avg_kl_oh=sum(r.avg_kl_oh for r in rs) / n,
                avg_jsd=sum(r.avg_jsd for r in rs) / n,
                fusion_win_rate=sum(r.fusion_win_rate for r in rs) / n,
                avg_fusion_gain=sum(r.avg_fusion_gain for r in rs) / n,
                oracle_rate=sum(r.oracle_rate for r in rs) / n,
                fused_entropy=sum(r.fused_entropy for r in rs) / n,
                ouro_tokens_used=sum(r.ouro_tokens_used for r in rs) / n,
                hrm_tokens_used=sum(r.hrm_tokens_used for r in rs) / n,
            ))

        tag = "speed_full"
        print(f"\nAveraged across {len(prompts)} prompts:\n")
        print(format_table(averaged, show_completions=args.show_completions))
        results = averaged

    if args.save:
        save_results(results, tag=tag)

    if args.graph:
        run_graph_benchmark(
            text=args.prompt,
            temperature=args.temp,
            cache=args.benchmark_cache,
            device=args.device,
            results_file=args.results_file,
        )


if __name__ == "__main__":
    raise SystemExit(main())
