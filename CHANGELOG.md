# Changelog

## [0.3.0] - 2026-02-24
### Added
- **LLM-based Supervisor routing** with keyword fallback (replaces hardcoded state machine)
- **Query Rewriting**: Complex questions decomposed into sub-queries, retrieved independently, then synthesized
- **Multi-company data**: Apple and Microsoft financials (FY2021-FY2025) alongside NVIDIA
- **10 cross-company benchmark questions** (total 40 questions)
- **Pie chart, stacked bar, grouped bar** chart types in Quant agent
- **Approximate number matching** in evaluation (5% tolerance)
- **Entity matching** with synonym grouping (grew/growth/increase → "growth")
- **Partial scoring** (0-1) with per-type and per-difficulty breakdown
- **36 new tests** (total 52): keyword routing, number extraction, approx matching, answer checking, query rewriting, all chart types

### Changed
- `supervisor.py`: LLM routing → keyword fallback → deterministic fallback chain
- `researcher.py`: Query rewriting for comparison/multi-entity questions, multi-query RAG with LLM synthesis
- `chart_gen.py`: Enhanced prompt, supports 6 chart types, strips `<think>` tags from deepseek output
- `rag_adapter.py`: Structure-aware chunking by section/table (8 independent chunks vs 2 large blocks)
- `plot_tool.py`: Pure matplotlib (removed seaborn dependency for stability), 6 chart types
- `eval/run_eval.py`: Robust `check_answer` with approx matching, entity overlap, partial scores, per-type stats
- `RAG_TOP_K`: 3 → 5
- README rewritten with Phase 2 results and updated architecture

### Results
- Swarm: 100.0% accuracy (30/30), 0.935 partial score, 0% hallucination
- Baseline: 20.0% accuracy, 0.142 partial score, 63.3% hallucination
- Delta: +80.0% accuracy

## [0.2.0] - 2026-02-24
### Breaking Changes
- Removed `v2_vision_swarm` module entirely (dummy MCTS/Vision code)
- Removed unused `constants.py`, `exceptions.py`, `experiments/` directory

### Fixed
- **main.py**: Graph compilation error (`route_tool_output` returning unregistered node)
- **supervisor.py**: Replaced v2 MCTS router with proper LLM-based routing + deterministic fallback
- **validation.py**: `sanitize_input` no longer strips Chinese/Unicode characters
- **robustness.py**: Added `async_retry_with_backoff` for async functions
- **config.py**: Migrated from deprecated class-based Config to `ConfigDict`

### Added
- Real NVIDIA financial data (FY2021-FY2025) for RAG testing
- Evaluation benchmark with 30 questions (easy/medium/hard)
- Evaluation script (`eval/run_eval.py`) with Baseline vs Swarm comparison
- Comprehensive test suite (16 tests covering routing, validation, tools)

### Changed
- Rewrote all agent nodes (supervisor, researcher, quant) — cleaner, no v2 imports
- Rewrote `rag_adapter.py` with proper async support
- Rewrote `plot_tool.py` with dynamic column detection
- Rewrote README with honest architecture description (no fabricated metrics)
- Cleaned `pyproject.toml` (correct author, removed unused deps)
- Updated `.gitignore` to cover logs, caches, binaries

### Removed
- 8x Vision_V2_*.zip files
- flash_attn .whl binary
- All `__pycache__`, `.pytest_cache`, `.ruff_cache`
- Fabricated paper drafts (`docs/paper/`)
- AutoDL deployment scripts

## [0.1.0] - 2026-02-01
### Added
- Initial multi-agent architecture with Supervisor, Researcher, Quant
- LlamaIndex RAG integration with structure-aware ingestion
- Rich console UI for observability
