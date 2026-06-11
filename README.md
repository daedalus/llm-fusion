# llm-fusion

Fused autoregressive text completion using ByteDance Ouro-1.4B + Sapient HRM-Text-1B.

Both models run under transformers 5.11.0. Fusion strategies operate in HRM's vocabulary space
via bidirectional token ID matching.

## Install

```bash
pip install -e ".[test]"
```

## Usage

```bash
# Fused (weighted average, default)
python -m llm_fusion --local "France's capital city is"

# Ouro only
python -m llm_fusion --model ouro --local "The first person to walk on the moon was"

# HRM only with chain-of-thought
python -m llm_fusion --model hrm --condition cot --local "Explain the sky is blue"

# CLI entry point
llm-fusion --local "Python was created by"
```

### Fusion Strategies

```bash
# Product of Experts — kills tokens either model dislikes
python -m llm_fusion --strategy product --local "The meaning of life is"

# Min-Entropy Routing — use the more confident model per token
python -m llm_fusion --strategy min-entropy --local "The capital of Japan is"

# Cascade — try Ouro first, fall back to HRM if Ouro's top prob < threshold
python -m llm_fusion --strategy cascade --cascade-threshold 0.5 --local "Explain quantum computing"

# Dynamic — Ouro weight linearly decays over generation steps
python -m llm_fusion --strategy dynamic --dynamic-initial-weight 0.8 --dynamic-final-weight 0.2 --local "Once upon a time"
```

### Perplexity Evaluation

```bash
# Evaluate a prompt's perplexity (lower = better)
python -m llm_fusion --model ouro --perplexity --local "The quick brown fox jumps over the lazy dog"
python -m llm_fusion --model hrm --perplexity --local "The quick brown fox jumps over the lazy dog"
python -m llm_fusion --perplexity --local "The quick brown fox jumps over the lazy dog"
```

### Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `-m` / `--model` | `fused` | Model: `fused`, `ouro`, `hrm` |
| `-n` / `--max-new-tokens` | `100` | Max tokens to generate |
| `--temp` / `--temperature` | `1.0` | Sampling temperature (`0` = greedy) |
| `--top-k` | `30` | Top-k tokens per model |
| `--threshold` | `0.01` | Min probability threshold |
| `--ouro-weight` | `0.5` | Ouro weight (average strategy) |
| `--rep-penalty` | `1.0` | Repetition penalty (`>1` discourages repeats) |
| `--condition` | `direct` | HRM condition: `direct`, `cot`, `noisy`, `synth` |
| `--strategy` | `average` | Fusion: `average`, `product`, `min-entropy`, `cascade`, `dynamic` |
| `--cascade-threshold` | `0.5` | Ouro top-prob threshold for cascade strategy |
| `--dynamic-initial-weight` | `0.8` | Starting Ouro weight for dynamic strategy |
| `--dynamic-final-weight` | `0.2` | Final Ouro weight for dynamic strategy |
| `--perplexity` | `false` | Evaluate perplexity instead of generating |
| `--local` | `false` | Load models from local directories |

## Fusion Strategies

| Strategy | Description |
|----------|-------------|
| `average` | Weighted average of Ouro and HRM logit distributions |
| `product` | Product of Experts — multiply probabilities, kills tokens either model dislikes |
| `min-entropy` | Per-token routing to the more confident model (lower entropy) |
| `cascade` | Try Ouro first; fall back to HRM if Ouro's top prob is below threshold |
| `dynamic` | Linear decay of Ouro weight from `initial` to `final` over generation steps |

## Requirements

- Python 3.11+
- transformers >=5.9.0 (HRM requirement)
- torch
- tokenizers
- `ouro-cache-fix` (optional, for KV cache)
- Ouro-1.4B and HRM-Text-1B model weights in `./Ouro-1.4B/` and `./HRM-Text-1B/`

## NaN Fixes for Ouro under transformers 5.x

Ouro-1.4B requires three patches to `modeling_ouro.py` under transformers 5.11.0:

1. **`_attn_implementation = None`** → set `config._attn_implementation = "eager"` before loading
2. **`torch.arange` on meta device** → pass `device="cpu"` to all `torch.arange` calls
3. **`inv_freq` buffer corruption** → recompute every forward via `rope_init_fn` closure instead of persistent buffer

See `AGENTS.md` for details.

## Project

```text
├── AGENTS.md                  # Agent instructions (NaN fixes, conventions)
├── CHANGELOG.md
├── LICENSE                    # Apache 2.0
├── docs/SPEC.md               # Specification
├── src/llm_fusion/            # Package
│   ├── __init__.py
│   ├── __main__.py            # python -m llm_fusion
│   ├── cli.py                 # CLI argument parsing
│   ├── generate.py            # Generation loop + perplexity evaluation
│   ├── fusion.py              # Fuser class (5 strategies)
│   ├── token_matcher.py       # Bidirectional token ID matcher
│   └── py.typed               # Type hints marker
├── tests/                     # Pytest suite (62+ tests)
├── Ouro-1.4B/                 # Model weights + patched modeling_ouro.py
└── HRM-Text-1B/               # Model weights
```
