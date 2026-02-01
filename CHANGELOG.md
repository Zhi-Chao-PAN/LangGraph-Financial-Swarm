# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-02-01
### Added
- **Phase 2 Optimization**: 20+ improvements for academic rigor.
- **Observability**: Rich console UI, Token estimation, Unique Run IDs.
- **Robustness**: Input sanitization, DataFrame validation decorator.
- **Docs**: ADR-001, CONTRIBUTING.md, and formalized LICENSE.

### Changed
- **Async Architecture**: Refactored `main.py` to use `asyncio`.
- **Config**: Migrated to `pathlib` for cross-platform safety.
- **Prompts**: Decoupled systems prompts to `src/prompts.py`.
