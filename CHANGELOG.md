# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-06-11

### Added
- Initial release
- Bidirectional token ID matcher between Ouro-1.4B and HRM-Text-1B
- Weighted-average logit fusion in HRM vocabulary space
- Autoregressive text completion (ouro/hrm/fused modes)
- HRM chat format with condition tags (direct/cot/noisy/synth)
- Ouro NaN fixes for transformers 5.11.0
- CLI entry point with --model, --condition, --temp, --rep-penalty flags
- Full test suite with 41+ tests

[0.1.0]: https://github.com/daedalus/LLM_EXPERIMENT/releases/tag/v0.1.0
