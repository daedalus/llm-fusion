# LLM Fusion — Agent Instructions

## Critical Ouro NaN Fixes (transformers 5.x / meta init context)

Ouro-1.4B (`modeling_ouro.py`) produces NaN under transformers 5.11.0. Three root causes and their fixes:

1. **`config._attn_implementation = None`** → `create_causal_mask` returns `None`
   - **Fix**: Set `config._attn_implementation = "eager"` before loading

2. **`torch.arange(...)` on meta device** — `torch.arange` creates meta tensors when running under transformers 5.x's `init_empty_weights` or similar meta-init context
   - **Fix**: Always pass `device="cpu"` to `torch.arange`

3. **`inv_freq` buffer corruption** — the buffer gets stale/zeroed during weight loading
   - **Fix**: Recompute `inv_freq` every forward via a `rope_init_fn` closure instead of caching it as a persistent buffer

Files to patch: `Ouro-1.4B/modeling_ouro.py` (lines ~434, ~514, ~630 in the patched version).

## Fused Mode Token Alignment

HRM tokenizes frequent words (e.g. "the") as character tokens (`t`, `h`, `e`) while Ouro uses subword tokens (` the`). To avoid drift:
- Re-encode Ouro `input_ids` from `generated_text` string each autoregressive step
- Do NOT map HRM token IDs to Ouro token IDs directly

## Test Suite

```bash
pip install -e ".[test]"
pytest -v --tb=short --cov=src
```

Coverage floor: 50% (many branches in `generate.py` require loading ~5GB model weights which isn't feasible in CI).

## Package Structure

```
src/llm_fusion/
  __init__.py       — version + re-exports
  __main__.py       — python -m entry point
  cli.py            — argparse CLI
  generate.py       — autoregressive generation loop, model patching, prompt formatting
  fusion.py         — weighted-average logit fusion logic
  token_matcher.py  — bidirectional token ID matcher (ouro ↔ hrm)
```

## Key Design Decisions

- `ouro_weight=0.5` balances Ouro (Universal Transformer, 4 UT steps, early-exit at 1.0) and HRM (prefix-LM, condition-tagged)
- HRM requires: `<|im_start|><|condition|>prompt<|im_end|>` format + `token_type_ids` = all-ones
- EOS tokens: Ouro = `<|endoftext|>` (id 0), HRM = `<|box_end|>` (id 11)
- Optional: `ouro-cache-fix==0.1.0` provides `UniversalTransformerCache` for KV cache
