# Amazon Review Analyzer v1.0.1 — System Overview

Amazon Review Analyzer is a local-first macOS review-intelligence application. It converts product reviews into traceable Aspect, Sentiment, Evidence, customer language, and deterministic business views while keeping the source review available for verification.

## Processing architecture

```text
Amazon Quick Scan / Paste Reviews / Upload Dataset
→ Validation
→ Cleaner
→ qwen3:8b through local Ollama
→ Aspect + Sentiment + Evidence
→ Aggregator
→ Keyword Analyzer
→ Business Intelligence
→ Dashboard
→ Saved Analysis
→ Competitor Comparison
→ Standalone Reports
```

Demo Dataset follows the same output contracts but is frozen and can be viewed without a live model.

## Responsibilities

| Layer | Responsibility |
| --- | --- |
| Input | Demo, Paste, CSV Upload, or compliant browser-assisted Amazon Quick Scan |
| Validation and Cleaner | Validate required fields, preserve originals, normalize only safe whitespace/HTML artifacts |
| Local analysis | Extract summary and Aspect-level sentiment with source Evidence |
| Runtime | Stable 1-worker execution, progress, checkpoint, Resume, and bounded provider retry |
| Aggregation | Review-level Aspect statistics, product vs shopping-experience separation, transparent counts and rates |
| Customer language | Business-relevant words and phrases grounded in completed reviews |
| Business Intelligence | Deterministic, sample-aware opportunities, priorities, and rating insights |
| Presentation | Bilingual Dashboard, Saved Analyses, A/B comparison, and standalone HTML reports |
| Desktop experience | First Run, Environment Health, Limited Mode, Diagnostics, launcher, install helper, and safe stop |

## Local data boundaries

- Reviews, checkpoints, outputs, saved analyses, comparisons, and reports stay in the local project workspace by default.
- The default provider is local Ollama with `qwen3:8b`; no OpenAI API key is required.
- Browsing Demo, switching language, opening saved results, comparing completed analyses, and exporting reports do not call a model.
- Amazon Quick Scan is the only input route that contacts Amazon. It respects login, CAPTCHA, verification, anti-bot, and session controls.
- Product analysis, Amazon analysis, and frozen Demo outputs are isolated so one mode does not overwrite another.

## Desktop architecture

`Amazon Review Analyzer.app` is a lightweight launcher, not a standalone runtime. It locates an approved local project, reuses the project Environment Health logic, starts or reuses one Streamlit instance, and opens `http://localhost:8501`.

The bundle does not contain Python, `.venv`, Ollama, `qwen3:8b`, or the complete backend. Setup and install helpers make that dependency visible rather than hiding it. Limited Mode keeps non-inference content available when local AI is unavailable.

## Reliability and safety

- Stable mode is intentionally serial at 1 worker.
- Resume does not reprocess completed or skipped reviews.
- Retry Failed Reviews targets only failed rows.
- Automatic retry is limited to one retry for explicit transient provider/connection errors.
- Evidence must pass strict source-substring validation.
- Saved Analysis and report sources are checked for schema/version compatibility and SHA-256 integrity.
- The Safe Stop helper validates project runtime identity before stopping Streamlit; it does not stop Ollama or unrelated Python processes.

## v1.0.1 scope

v1.0.1 is a Maintenance / UX / Distribution Release. It formalizes the accepted desktop experience and does not change the model, Prompt, taxonomy, sentiment logic, Evidence validation, aggregation, keyword analysis, Amazon strategy, business metrics, comparison, reports, or frozen evaluations.
