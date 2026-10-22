# Changelog

All notable product-level changes for Amazon Review Analyzer are documented here. This file summarizes shipped capability rather than internal development milestones.

## [1.0.0] — 2026-10-16

### Added

- Local aspect-based review analysis with Ollama and `qwen3:8b` as the default model stack.
- Structured Summary, canonical Aspect, sentiment, and continuous-substring Evidence output.
- Fifteen canonical aspects covering product, usage, shopping experience, and Other.
- Four input modes: Demo Dataset, Amazon Quick Scan, CSV Upload, and Paste Reviews.
- Validation and cleaning that preserve original review content and identify non-analyzable rows.
- Review-level deduplication for customer-count aggregation while retaining mention-level Evidence.
- Product selling points, pain points, customer concerns, shopping-experience insights, opportunity ranking, priority matrix, and rating-based insights.
- Customer-language analysis with useful unigram, bigram, and trigram phrases.
- Five Dashboard views: Overview, Product Insights, Customer Language, Aspect Deep Dive, and Review Explorer.
- English and Simplified Chinese interface with original customer language preserved.
- Persistent Saved Analyses with stable IDs, metadata, and integrity validation.
- Deterministic Product A vs Product B comparison from compatible saved analyses.
- English and Chinese standalone HTML reports for single-product and competitor-comparison results.
- macOS one-click launcher and Demo-first startup behavior.

### Reliability

- Stable single-worker production runtime selected from the frozen throughput benchmark.
- Row-level checkpointing, Resume, progress counters, elapsed time, and dynamic ETA.
- Retry Failed Reviews without reprocessing successful or skipped rows.
- One bounded retry for recognized transient provider/connection failures; no unbounded retry behavior.
- Graceful handling for individual review failures, missing local AI, unavailable browser runtime, and Amazon access limitations.
- Dataset isolation between frozen Demo, current Upload/Paste analysis, and current Amazon analysis.
- Snapshot and release-asset SHA-256 validation.
- Frozen evaluation framework and protected model/taxonomy contracts.

### Amazon access

- Normal visible-browser import flow with supported Amazon marketplace and URL normalization.
- Manual login or human-verification continuation without bypassing Amazon access controls.
- Distinct user-facing states for access limits, no reviews, page changes, invalid URLs, connection issues, and TLS issues.
- Paste Reviews and CSV Upload fallback paths when live Amazon access is unavailable.

### Security and privacy

- Local model inference by default; no OpenAI API key required for the v1.0 default configuration.
- HTML escaping for dynamic report content.
- Local-only Saved Analysis, comparison, and report storage by default.
- Minimal runtime event logging that avoids storing full review text in event records.
- Release packaging rules that exclude secrets, cookies, browser profiles, private logs, `.env`, `.venv`, caches, and model files.
- Report and documentation checks for local absolute-path leakage.

### Validation

- Product Release Acceptance: **PASS**.
- Automated tests: **1077 / 1077 passed**.
- Open release issues: **0 BLOCKER / 0 MAJOR**.
- Frozen Blind V2: 150 cases; Aspect + Sentiment F1 **85.66%**; Aspect-only F1 **87.11%**; Sentiment Accuracy **98.33%**; Evidence Validity and Evidence Match **100%**; Strict Exact Case Accuracy **66.67%**.

The frozen evaluation results describe that specific evaluation set and configuration; they are not universal accuracy claims.

### Known limitations

- Live Amazon review availability depends on Amazon access, login, human verification, marketplace behavior, and page structure.
- Local-model latency depends on hardware and workload.
- Small samples and business scores are directional decision-support signals, not statistical conclusions.
- Competitor Comparison supports two products in v1.0.
- The application is local-first and does not provide cloud accounts or shared team storage.

[1.0.0]: RELEASE_NOTES.md
