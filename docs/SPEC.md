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
- Single-step fusion of next-token logits via weighted average
- Autoregressive text completion (ouro | hrm | fused modes)
- HRM: chat-format prompts, condition tags (direct/cot/noisy/synth), prefix-LM mask
- Ouro: patched modeling_ouro.py (3 NaN fixes under transformers 5.11.0)
- Repetition penalty, temperature scaling, top-k sampling
- CLI entry point with `--model`, `--condition`, `--temp`, `--rep-penalty` flags

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
    ) -> None
    def fuse_logits(
        self, ouro_logits: list[float], hrm_logits: list[float]
    ) -> list[tuple[int, float, str]]
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
) -> None
```

### CLI

```bash
python -m llm_fusion [--model ouro|hrm|fused] [--local]
    [--temp FLOAT] [--top-k INT] [--rep-penalty FLOAT]
    [--condition direct|cot|noisy|synth] [--ouro-weight FLOAT]
    "Your prompt here"
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

## Performance & Constraints

- Both models load simultaneously in fused mode (~2.7GB + ~2.3GB)
- Step-by-step generation is O(n) forward passes (no generate() due to cache issues)
- Ouro-1.4B: GPT-2 tokenizer (49,152 vocab), 1.43B params
- HRM-Text-1B: Qwen2 tokenizer (65,536 vocab), 1.18B params
- Shared tokens: ~37,306
- transformers >=5.9.0 required (HRM constraint)
- Ouro requires trust_remote_code=True with patched modeling_ouro.py
