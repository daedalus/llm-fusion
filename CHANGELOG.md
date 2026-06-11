# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-06-11

### Added
- Initial release
- Bidirectional token ID matcher between Ouro-1.4B and HRM-Text-1B
- Weighted-average logit fusion in HRM vocabulary space
- 5 fusion strategies: average, product, min-entropy, cascade, dynamic
- Autoregressive text completion (ouro/hrm/fused modes)
- HRM chat format with condition tags (direct/cot/noisy/synth)
- Ouro NaN fixes for transformers 5.11.0
- CLI entry point with --model, --condition, --temp, --rep-penalty flags
- KL divergence measurement between model distributions (--kl)
- Fusion gain per-token display (--gain)
- Evaluation mode: score reference text under all 3 configurations (--eval)
- Fusion quality metrics module: fusion_gain, evaluate_text, compare_distributions
- Speed benchmarks for all model/strategy combinations
- Robustness benchmark: 26 prompts across 8 categories
- Perplexity evaluation for single-model and fused modes
- Full test suite with 96+ tests

[0.1.0]: https://github.com/daedalus/LLM_EXPERIMENT/releases/tag/v0.1.0
