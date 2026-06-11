# SPEC.md — LLM Fusion

## Purpose

Fuse two transformer LMs (ByteDance Ouro-1.4B + Sapient HRM-Text-1B) for
autoregressive text completion. Both models run under the same transformers
version (5.11.0). HRM requires a chat-format prompt with condition tags
and prefix-LM `token_type_ids`. Ouro requires a patched `modeling_ouro.py`
to fix NaN under 5.11.0. Fusion is a weighted average of logits in HRM's
vocabulary space via bidirectional token ID matching.

## Scope

### In scope
- Token-level bidirectional matching between Ouro and HRM vocabularies
- Single-step fusion of next-token logits via weighted average (5 strategies: average, product, min-entropy, cascade, dynamic)
- Autoregressive text completion (ouro | hrm | fused modes)
- HRM: chat-format prompts, condition tags (direct/cot/noisy/synth), prefix-LM mask
- Ouro: patched modeling_ouro.py (3 NaN fixes under transformers 5.11.0)
- Repetition penalty, temperature scaling, top-k sampling
- CLI entry point with --model, --condition, --temp, --rep-penalty, --strategy flags
- KL divergence measurement between model distributions (--kl)
- Fusion gain: per-token log-ratio of fused vs best parent probability (--gain)
- Evaluation mode: score reference text under all 3 configurations (--eval)
- Fusion quality metrics: fusion_gain, evaluate_text, compare_distributions
- Speed benchmarks for all model/strategy combinations
- Robustness benchmark: 26 diverse prompts across 8 categories
- Perplexity evaluation for single-model and fused modes

### Not in scope
- Training or fine-tuning either model
- GPU/CUDA optimization (basic device_map support only)
- Model downloading or management
- Streaming generation
- batch inference

## Public API / Interface

### `llm_fusion.token_matcher`

```python
class Match:
    confidence: str       # "exact" | "approx" | "mismatch" | "invalid"
    target_ids: list[int]
    source_str: str | None
    target_str: str | None
    note: str

class TokenMatcher:
    def __init__(self) -> None
    def ouro_to_hrm(self, token_id: int) -> Match
    def hrm_to_ouro(self, token_id: int) -> Match
    def map_sequence(self, token_ids: list[int], src: str) -> Match
    def format_match(self, m: Match, src_name: str, src_id: int | None = None) -> str
    def show_info(self) -> None
```

### `llm_fusion.fusion`

```python
class Fuser:
    def __init__(
        self, matcher: TokenMatcher,
        ouro_tok: Tokenizer, hrm_tok: Tokenizer,
        ouro_weight: float = 0.5,
        top_k: int = 50, threshold: float = 0.01,
        strategy: str = "average",
        cascade_threshold: float = 0.5,
        dynamic_initial_weight: float = 0.8,
        dynamic_final_weight: float = 0.2,
        dynamic_total_steps: int = 100,
    ) -> None
    def fuse_logits(
        self, ouro_logits: list[float], hrm_logits: list[float]
    ) -> list[tuple[int, float, str]]
    def sample_token(
        self, ouro_logits: list[float], hrm_logits: list[float],
        temperature: float = 1.0,
    ) -> tuple[int, str, float]
    def model_distributions(
        self, ouro_logits: list[float], hrm_logits: list[float]
    ) -> tuple[dict[int, float], dict[int, float]]
```

### `llm_fusion.generate`

```python
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
    cascade_threshold: float = 0.5,
    dynamic_initial_weight: float = 0.8,
    dynamic_final_weight: float = 0.2,
    perplexity: bool = False,
    show_kl: bool = False,
    show_gain: bool = False,
    eval_text: str = "",
) -> None
```

### Fusion Strategies

| Strategy | Behavior |
|----------|----------|
| `average` | Weighted average: `p = ouro_weight * p_ouro + (1-ouro_weight) * p_hrm` |
| `product` | Product of Experts: `p ∝ p_ouro * p_hrm` — kills tokens either model dislikes |
| `min-entropy` | Per-token routing to the more confident model (lower distribution entropy) |
| `cascade` | Try Ouro first; fall back to HRM if Ouro's top prob < threshold |
| `dynamic` | Ouro weight linearly decays from `initial` to `final` over generation steps |

### Fusion Quality Metrics

```python
def fusion_gain(fused_prob: float, ouro_prob: float, hrm_prob: float) -> float
    # log P_fused(token) - max(log P_ouro(token), log P_hrm(token))
    # positive means fusion beats the best parent

def evaluate_text(text, ouro_model, hrm_model, ouro_tok, hrm_tok, fuser, device, max_tokens) -> dict
    # Scores reference text under all 3 configs, returns fusion gain, win rate, PPL

def compare_distributions(ouro_logits, hrm_logits, ...) -> dict
    # Compares Ouro vs HRM distributions: entropy, overlap, KL divergence
```

### Benchmark

```python
def run_benchmark(text, max_new_tokens, ...) -> list[BenchmarkResult]
    # Speed benchmark for all model/strategy combos

def run_robustness_benchmark(max_new_tokens, ...) -> list[RobustnessResult]
    # Diverse battery (26 prompts, 8 categories): PPL, gain, KL, entropies
```

### CLI

```bash
python -m llm_fusion [--model ouro|hrm|fused] [--local]
    [--temp FLOAT] [--top-k INT] [--rep-penalty FLOAT]
    [--condition direct|cot|noisy|synth] [--ouro-weight FLOAT]
    [--strategy average|product|min-entropy|cascade|dynamic]
    [--cascade-threshold FLOAT] [--perplexity] [--kl] [--gain]
    [--eval TEXT]
    "Your prompt here"

python -m llm_fusion benchmark [--prompt TEXT] [-n TOKENS] [--robustness]
```

## Data Formats

- Tokenizer files: HuggingFace `tokenizer.json` (SentencePiece / BPE)
- Model weights: HuggingFace `model.safetensors`
- HRM prompt format: `<|im_start|><|{condition}|>prompt<|im_end|>`
- HRM EOS token: `<|box_end|>` (id=11)
- Ouro EOS token: id=0

## Edge Cases

1. Empty prompt string → argparse rejects (nargs="?")
2. Single-token completion → fused mode may generate EOS immediately
3. Repetition penalty > 1 with seen_ids from both models' token spaces
4. Ouro config._attn_implementation=None → must be set to "eager"
5. HRM prompt encoding differs from raw text (chat format prefix/suffix)
6. Token IDs not in vocabulary → Match confidence="invalid"
7. Temperature=0 → greedy argmax (bypasses sampling)
8. model="ouro" with HRM imports → load_ouro=True, load_hrm=False
9. model="hrm" with Ouro imports → reversed
10. fused mode: Ouro input_ids reconstructed from generated text each step
11. Zero logits list → softmax_top_k returns empty lists
12. Negative logit values in KL computation → clamped to 1e-10
13. Empty eval_text → skipped (returns empty results)
14. Eval mode with empty fuser argument → handled via conditional
15. HRM output decoding with strip of chat format markers
16. Cascade strategy: Ouro top prob equals threshold exactly → uses Ouro

## Performance & Constraints

- Both models load simultaneously in fused mode (~2.7GB + ~2.3GB)
- Step-by-step generation is O(n) forward passes (no generate() due to cache issues)
- Ouro-1.4B: GPT-2 tokenizer (49,152 vocab), 1.43B params
- HRM-Text-1B: Qwen2 tokenizer (65,536 vocab), 1.18B params
- Shared tokens: ~37,306
- transformers >=5.9.0 required (HRM constraint)
- Ouro requires trust_remote_code=True with patched modeling_ouro.py
