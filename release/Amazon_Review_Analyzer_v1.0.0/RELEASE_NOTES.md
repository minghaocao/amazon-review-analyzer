# Amazon Review Analyzer v1.0.0 — Release Notes

| Release field | Value |
| --- | --- |
| Release type | Stable Release |
| Product release acceptance | **PASS** |
| Automated tests | **1077 / 1077 passed** |
| BLOCKER issues | **0** |
| MAJOR issues | **0** |
| Default provider | Ollama |
| Default model | `qwen3:8b` |
| Runtime mode | Stable, 1 worker |
| UI languages | English / 简体中文 |
| Release date | 2026-10-16 |

## Release summary

Amazon Review Analyzer v1.0.0 is the first stable product release of a local-first review intelligence application. It converts customer review text into normalized aspects, aspect-level sentiment, validated source Evidence, keywords, and transparent business insights. Completed analyses can be saved, compared across two products, and exported as standalone HTML reports.

This is a packaging and presentation release of the accepted v1.0 product. The release does not change the frozen model prompt, canonical taxonomy, evaluator definitions, business scoring semantics, Amazon access behavior, or stable runtime policy.

## Included product capabilities

### Input and preparation

- Frozen Demo Dataset for offline demonstration.
- Paste Reviews workflow with one non-empty review per line.
- CSV Upload with column mapping, preview, validation, and a 10 MB upload limit.
- Amazon Quick Scan through a normal visible browser session, with explicit handling of login, human verification, access limits, no reviews, and page-structure changes.
- Raw review preservation, basic cleaning, analyzability status, and structured data validation.

### Local AI analysis

- Default local inference through Ollama and `qwen3:8b`.
- Aspect-Based Sentiment Analysis rather than one sentiment label per review.
- Fifteen fixed canonical aspects spanning product quality, usage experience, and shopping experience.
- Sentiment and Evidence at the individual aspect-mention level.
- Strict Evidence validation against continuous substrings of the original review.
- Review-level aspect deduplication for customer-count aggregation while retaining original mention-level Evidence.

### Business intelligence and exploration

- Customer concerns, product selling points, product pain points, and shopping-experience insights.
- Opportunity Ranking and Customer Priority Matrix.
- Rating insights, hidden complaints in positive-rated reviews, and strengths found in negative-rated reviews.
- Deterministic business recommendations with visible sample-support labels.
- Commercially useful unigram, bigram, and trigram analysis.
- Five Dashboard views: Overview, Product Insights, Customer Language, Aspect Deep Dive, and Review Explorer.

### Productization

- English and Simplified Chinese interface without reanalysis.
- Stable one-worker runtime with row-level checkpoints, Resume, elapsed time, dynamic ETA, and targeted retry.
- Persistent Saved Analysis library with version and SHA-256 integrity validation.
- Deterministic two-product comparison with customer Evidence and sample caveats.
- English and Chinese standalone HTML reports for single-product and competitor-comparison outputs.
- macOS one-click launcher support.

## Release validation

### Product release gate

- Automated Tests: **1077 / 1077 passed**
- Product Release Acceptance: **PASS**
- BLOCKER: **0**
- MAJOR: **0**

The accepted release gate covers offline regression tests, frozen-asset integrity, bilingual UI contracts, stable runtime behavior, saved-analysis integrity, deterministic comparison, report safety, and release-health checks. The normal offline suite uses mocks and local fixtures; it does not require live Amazon, OpenAI, or Ollama access.

### Frozen Blind V2 model evaluation

| Metric | Frozen result |
| --- | ---: |
| Cases | 150 |
| Aspect + Sentiment Precision | 84.29% |
| Aspect + Sentiment Recall | 87.08% |
| Aspect + Sentiment F1 | **85.66%** |
| Aspect-only Precision | 85.71% |
| Aspect-only Recall | 88.56% |
| Aspect-only F1 | **87.11%** |
| Sentiment Accuracy | **98.33%** |
| Evidence Validity | **100%** |
| Evidence Match | **100%** |
| Strict Exact Case Accuracy | **66.67%** (100 / 150) |

The frozen run produced successful structured predictions for 147 of 150 cases. Evaluation artifacts are frozen and are not rerun as part of routine startup, packaging, or offline release validation.

> **Interpretation:** These numbers describe one frozen 150-case Blind V2 set evaluated with the documented `qwen3:8b` configuration. They are not a universal system accuracy, a statistical confidence interval, or a guarantee for every product, language, review style, or real-world dataset. Evidence percentages apply only to Evidence produced and scored under that evaluation protocol.

### Frozen performance benchmark

The 20-review throughput-and-stability benchmark selected Stable mode with one worker:

| Measure | Frozen result |
| --- | ---: |
| Successful / attempted | 20 / 20 |
| Total runtime | 294.67 seconds |
| Median per-review latency | 12.14 seconds |
| P90 per-review latency | 24.05 seconds |
| Throughput | 4.07 reviews/minute |
| Retries / model failures | 0 / 0 |

The benchmark evaluates runtime behavior, not model accuracy. Results were produced on one accepted local environment and are not a speed guarantee. Hardware, review length, model loading state, and background workload can materially change performance. Two-worker mode was not selected because it reduced measured throughput and increased latency in that environment.

## Data, privacy, and security posture

- v1.0 uses local Ollama inference by default and does not require an OpenAI API key.
- Original reviews and generated outputs remain in the local project workspace by default.
- Analysis Library snapshots and exported reports are local files; there is no cloud account or automatic upload.
- Amazon Quick Scan is the only mode that visits Amazon and is subject to current site accessibility.
- Dynamic report content is HTML escaped.
- Versioned snapshots and protected release assets use SHA-256 integrity checks.
- This release package excludes `.env`, credentials, cookies, browser profiles, private logs, `.venv`, caches, and Ollama model files.

## Known limitations

1. **Amazon availability is not guaranteed.** Amazon may require login or human verification, limit automated access, change page structure, or expose fewer reviews than a product page's displayed total. The application does not bypass those controls.
2. **Full review retrieval cannot be guaranteed.** Quick Scan analyzes only reviews accessible in the current visible browser session. Use CSV Upload or Paste Reviews for controlled datasets.
3. **Local performance depends on hardware.** `qwen3:8b` inference time varies with memory, processor/GPU, model warm state, review length, and concurrent workloads.
4. **Small samples are directional.** A few reviews can be analyzed, but their rankings and rates should not be treated as statistically representative.
5. **Business scores are decision-support heuristics.** Selling points, pain points, opportunity scores, and comparison labels are transparent directional metrics—not significance tests, causal findings, market-share estimates, or automatic decisions.
6. **Comparison is limited to two products.** v1.0 supports Product A vs Product B from compatible saved analyses; it does not provide multi-product portfolio comparison.
7. **Data comparability remains a human responsibility.** Products should be compared using reasonably comparable sources, time windows, and sample sizes.
8. **The canonical taxonomy is fixed.** Category-specific attributes that do not map cleanly to one of the 15 categories may be retained as Other.
9. **Customer language is preserved.** Switching the UI language does not translate original reviews, Evidence, or extracted customer phrases.
10. **The app is local-first.** v1.0 has no cloud deployment, user accounts, team workspace, shared database, or automatic synchronization.
11. **Reports are HTML-first.** Reports can be printed or saved as PDF through a browser, but the product does not create native PDF files.

## Installation and package boundaries

The release portfolio package does not bundle the Python virtual environment, Playwright browser runtime, Ollama application, or `qwen3:8b` model. These must be installed separately according to the Quick Start documentation. This keeps the package portable and avoids distributing large or machine-specific artifacts.

## Documentation

- [Release README](README.md)
- [中文快速开始](QUICK_START_CN.md)
- [English Quick Start](QUICK_START_EN.md)
- [System Overview](SYSTEM_OVERVIEW.md)
- [中文完整系统说明](docs/Amazon_Review_Analyzer_v1.0_User_Guide_CN.md)
- [Portfolio Summary](PORTFOLIO_SUMMARY.md)
- [Changelog](CHANGELOG.md)

## License status

No software license is currently included. The absence of a license must not be interpreted as permission to copy, distribute, or modify the source code.
