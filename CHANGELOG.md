# Changelog

All notable release-level changes to Amazon Review Analyzer are documented here. Historical release artifacts remain immutable in their original release folders.

## v1.0.1

**Type:** Maintenance / UX / Distribution Release

### Added

- Native-style macOS `.app` launcher
- First-run onboarding in English and Simplified Chinese
- Environment Health Check and machine-readable health output
- Advanced Diagnostics with privacy-safe details
- Limited Mode for viewing Demo, saved results, comparisons, reports, and documentation without local AI
- Install-to-Applications helper
- Safe Stop helper that targets only the validated project Streamlit process

### Improved

- startup UX and recovery messaging
- local environment visibility before new analysis
- launcher safety and single-instance behavior
- bounded project discovery for the local launcher
- bilingual setup, onboarding, health, and diagnostics guidance

### Unchanged

- local AI provider and `qwen3:8b` model
- Prompt, taxonomy, sentiment logic, Evidence validation, and post-processing
- Aggregator and Keyword Analyzer
- Amazon access strategy and compliance boundaries
- Business Intelligence, competitor comparison, and report semantics
- all frozen datasets, predictions, evaluation results, performance baselines, Demo outputs, saved analyses, comparisons, and reports

## v1.0.0

Initial stable release with local evidence-grounded review analysis, aggregation, customer-language analysis, bilingual Dashboard, Saved Analyses, two-product comparison, standalone HTML reports, Amazon browser-assisted Quick Scan, and stable checkpoint/resume/retry runtime.

The complete historical v1.0.0 notes, manifest, hashes, and package remain in the original v1.0.0 release assets and are not rewritten by v1.0.1.
