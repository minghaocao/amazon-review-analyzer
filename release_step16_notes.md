# Step 16 — Installation & First-Run Experience Notes

Product: **Amazon Review Analyzer**  
Current product version: **v1.0.0**  
Recommended future maintenance release: **v1.0.1**

## Purpose

Step 16 improves installation, environment health, first-run guidance, Limited Mode, diagnostics, localization, and user-facing recovery messages. It does not change review-analysis semantics or introduce a new product feature set.

## Installation experience

The source project now includes:

- `setup_amazon_review_analyzer.command` for explicit first-time setup on macOS;
- `QUICK_START_CN.md` and `QUICK_START_EN.md` as current living setup guides;
- documented `python main.py --health-check` and `--health-check-json` workflows;
- clear Ready, Action Required, Fatal, and Limited Mode behavior;
- recovery guidance for Python, dependencies, Ollama, `qwen3:8b`, port conflicts, interrupted analysis, and Resume.

The setup script may create a missing `.venv`, install the existing `requirements.txt`, and create `.env` from `.env.example` only when `.env` does not exist. It never deletes an existing environment or overwrites user configuration.

## Explicit safety boundaries

Setup and environment health do not:

- install Ollama silently;
- download `qwen3:8b` automatically;
- access Amazon;
- call OpenAI;
- run review inference;
- change Stable 1-worker runtime behavior;
- delete user data, checkpoints, snapshots, or reports;
- reset or clean the repository.

Limited Mode keeps Demo Dataset, Saved Analyses, saved comparisons, reports, and documentation available when local AI is unavailable. New analysis remains gated until the required local environment is ready.

## Frozen release protection

The Step 15 release package remains frozen. Step 16 must not overwrite or rebuild:

- `release/Amazon_Review_Analyzer_v1.0.0.zip`;
- files inside `release/Amazon_Review_Analyzer_v1.0.0/`;
- Demo frozen outputs;
- Blind V2, Final Blind, Holdout, or Dev V2 assets;
- model-core files and frozen evaluation reports;
- Saved Analysis snapshots, performance assets, comparison outputs, or exported reports.

The root Quick Start and User Guide are living documentation for the source project. They are intentionally separate from the frozen Step 15 portfolio package.

## Model and business logic

Step 16 does not modify:

- the LLM prompt or default `qwen3:8b` configuration semantics;
- Aspect taxonomy semantics;
- review analysis or Evidence validation logic;
- evaluator metric definitions;
- Aggregator, Keyword Analyzer, Business Intelligence, or Competitor Comparison semantics;
- Amazon access strategy;
- Stable 1-worker runtime policy.

Frozen evaluations must not be rerun as part of Step 16 acceptance. Integrity is checked by SHA-256 instead.

## Version recommendation

Because Step 16 changes installation and first-run reliability without changing model semantics or business behavior, a future distributable update should be considered **v1.0.1**, not v1.1.0.

This note does not change the current application version and does not create a new release package.

