# llm-fusion

Fused autoregressive text completion using ByteDance Ouro-1.4B + Sapient HRM-Text-1B.

Both models run under transformers 5.11.0. Fusion is a weighted average of logits
in HRM's vocabulary space via bidirectional token ID matching.

## Install

```bash
pip install -e ".[test]"
```

## Usage

```bash
# Fused (default)
python -m llm_fusion --local "France's capital city is"

# Ouro only
python -m llm_fusion --model ouro --local "The first person to walk on the moon was"

# HRM only with chain-of-thought
python -m llm_fusion --model hrm --condition cot --local "Explain the sky is blue"

# CLI entry point
llm-fusion --local "Python was created by"
```

## Requirements

- Python 3.11+
- transformers >=5.9.0 (HRM requirement)
- torch
- tokenizers
- `ouro-cache-fix` (optional, for KV cache)
- Ouro-1.4B and HRM-Text-1B model weights in `./Ouro-1.4B/` and `./HRM-Text-1B/`

## Project

```text
├── docs/SPEC.md              # Specification
├── src/llm_fusion/           # Package
│   ├── __init__.py
│   ├── __main__.py           # python -m llm_fusion
│   ├── cli.py                # CLI argument parsing
│   ├── generate.py           # Generation loop
│   ├── fusion.py             # Fuser class
│   ├── token_matcher.py      # TokenMatcher class
│   └── py.typed              # Type hints marker
├── tests/                    # Pytest suite
├── Ouro-1.4B/                # Model weights + patched modeling_ouro.py
└── HRM-Text-1B/              # Model weights
```
