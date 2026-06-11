__version__ = "0.1.0"
__all__ = [
    "Match",
    "TokenMatcher",
    "Fuser",
    "CausalLM",
    "load_models",
    "patch_ouro_model",
    "generate",
    "format_hrm_prompt",
    "strip_hrm_output",
    "compute_perplexity",
    "compute_fused_perplexity",
    "compute_kl",
    "fusion_gain",
    "evaluate_text",
    "compare_distributions",
    "BenchmarkResult",
    "run_benchmark",
    "OURO_EOS_ID",
    "HRM_EOS_ID",
]

from llm_fusion.benchmark import BenchmarkResult, run_benchmark
from llm_fusion.fusion import Fuser, compute_kl
from llm_fusion.generate import (
    HRM_EOS_ID,
    OURO_EOS_ID,
    compute_fused_perplexity,
    compute_perplexity,
    format_hrm_prompt,
    generate,
    strip_hrm_output,
)
from llm_fusion.loader import CausalLM, load_models, patch_ouro_model
from llm_fusion.metrics import compare_distributions, evaluate_text, fusion_gain
from llm_fusion.token_matcher import Match, TokenMatcher
