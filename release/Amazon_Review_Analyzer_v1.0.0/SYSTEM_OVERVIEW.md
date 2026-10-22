# Amazon Review Analyzer v1.0.0 — System Overview

This document is a short technical and product orientation. For installation and operation, use the [Quick Start](QUICK_START_EN.md); for the complete Chinese reference, use the [中文系统说明](docs/Amazon_Review_Analyzer_v1.0_User_Guide_CN.md).

## Purpose

Amazon Review Analyzer converts unstructured customer reviews into traceable product intelligence. Its central contract is simple:

> Every business-facing conclusion should remain connected to structured Aspect + Sentiment data and the original customer Evidence that supports it.

The product is local-first. Its accepted v1.0 default uses Ollama with `qwen3:8b`, a stable one-worker runtime, local outputs, and an English / Simplified Chinese Streamlit interface.

## End-to-end architecture

```text
Input
  Demo | Paste | CSV Upload | Amazon Quick Scan
                         ↓
Validation and Cleaning
  schema checks | original-text preservation | analyzability
                         ↓
Local LLM Analysis
  summary | raw aspect | canonical aspect | sentiment | Evidence
                         ↓
Validation and Runtime Control
  schema | Evidence substring | retry | checkpoint | Resume
                         ↓
Deterministic Post-processing
  review-level deduplication | aggregation | keywords | BI
                         ↓
Interactive Product Views
  Overview | Product Insights | Customer Language
  Aspect Deep Dive | Review Explorer
                         ↓
Persistent and Portable Outputs
  Saved Analysis → A/B Comparison → Standalone Reports
```

## Major components

| Layer | Responsibilities |
| --- | --- |
| Input | Demo, pasted text, mapped CSV, and browser-assisted Amazon review import |
| Validation | Required-field checks, rating diagnostics, duplicate IDs, and safe handling of missing content |
| Cleaning | Whitespace and HTML cleanup while retaining original text, case, punctuation, numbers, and negation |
| AI analysis | Review summary and aspect-level sentiment/Evidence extraction through the configured provider |
| Taxonomy | Normalization into 15 fixed canonical aspects |
| Evidence control | Continuous-substring validation against the original review; invalid Evidence cannot silently support a result |
| Batch runtime | Stable one-worker execution, per-row status, checkpoint, Resume, bounded retry, and failed-row retry |
| Aggregation | Review-level counts and rates without treating repeated mentions in one review as multiple customers |
| Customer language | Offline keyword and phrase statistics with review references |
| Business intelligence | Deterministic rankings, sample-support labels, priority matrix, rating insights, and recommendations |
| Presentation | Bilingual Dashboard, Analysis Library, two-product comparison, and standalone HTML reports |

## Input contracts

### Demo Dataset

The frozen Demo is the default view. It contains 15 reviews, 14 successfully analyzed rows, one skipped row, and 93.3% coverage. It is designed to demonstrate the complete product without live network or model access.

### Paste Reviews

Each non-empty line becomes one review. Stable IDs are generated, and the user reviews a preview before starting local AI analysis.

### CSV Upload

The UI requires a column mapped to review content. Review ID, rating, title, date, and verified-purchase status are optional mappings; missing IDs can be generated. The interactive upload limit is 10 MB.

### Amazon Quick Scan

The product opens a normal visible browser and analyzes only reviews accessible in that browser session. Login and human verification remain user actions. The importer does not bypass CAPTCHA, Robot Check, Access Denied, or other Amazon controls, and it cannot guarantee full review retrieval.

## Analysis contract

Each analyzable review can produce:

- a concise summary;
- one or more aspect mentions;
- the original aspect phrase and its normalized canonical aspect;
- `positive`, `negative`, or `neutral` sentiment for each mention;
- an Evidence fragment copied from the original review.

The pipeline preserves mention-level results and derives review-level unique aspect/sentiment records for aggregation. If the same review repeats the same aspect and sentiment three times, it counts as one customer-level record while retaining a mention count of three and all original Evidence. Conflicting positive and negative mentions are not silently collapsed.

At aggregate level, a review/aspect combination may be represented as mixed when positive and negative signals coexist. This preserves conflict without pretending that the customer expressed only one direction.

## Canonical taxonomy

### Product aspects

Picture Quality, Sound Quality, Price / Value, Remote Control, System Performance, Ease of Use, Design, Build Quality, Reliability, Connectivity, and Setup / Installation.

### Shopping-experience aspects

Delivery, Packaging, and Customer Service.

### Other

Other is retained for auditability but excluded from the default primary commercial rankings.

## Deterministic intelligence layer

After AI extraction, aggregation, keyword statistics, business scoring, comparison, and report generation are deterministic. They do not require a second model call.

Key semantics:

- **Customer Concern** reflects how often an aspect appears in successful analyzed reviews.
- **Selling Point** requires a product aspect with sufficient sample support and more positive than negative reviews.
- **Pain Point** requires a product aspect with sufficient sample support and more negative than positive reviews.
- **Shopping Experience** keeps Delivery, Packaging, and Customer Service separate from product-quality rankings.
- **Opportunity Score** combines review share and negative rate as a directional prioritization aid.
- **Evidence Level** describes sample support; it is not a statistical confidence level.
- **Priority Matrix** uses review share and net sentiment to organize Strength, Improvement Priority, Minor Positive, Emerging Concern, and Balanced/Monitor signals.

All of these outputs are decision-support heuristics. They do not establish statistical significance or causation.

## Stable runtime

v1.0 production analysis uses one worker. The accepted frozen 20-review benchmark showed lower throughput with two workers on the evaluated 16 GiB Apple Silicon environment, so concurrent local model requests are not enabled as the default.

The runtime provides:

- a checkpoint after each processed row;
- status values `success`, `failed`, and `skipped`;
- elapsed time, counters, and dynamic ETA after enough timing data exists;
- Resume without repeating compatible `success` or `skipped` rows;
- Retry Failed Reviews without reprocessing completed rows;
- at most one automatic retry for a recognized transient provider/connection failure.

Evidence, schema, content-quality, and user-input errors are not automatically retried.

## Data isolation and persistence

The product keeps three active-data contexts separate:

```text
Frozen Demo outputs
Current Upload / Paste analysis
Current Amazon Quick Scan analysis
```

Completing one workflow does not overwrite the others. A completed result can be saved into the Analysis Library as a versioned local snapshot containing the Dashboard's required output files, metadata, and integrity hashes.

Before a snapshot is loaded, compared, or used to generate a report, the product checks schema compatibility, analysis/taxonomy identity, required files, and hashes. Integrity failure stops the operation rather than silently trusting altered data.

## Comparison and reporting

Competitor Comparison loads exactly two different compatible saved analyses. It compares the same canonical aspects, ratings, customer language, and Evidence and adds sample-size warnings. Missing Evidence is displayed as no evidence, not as a zero-positive result. The comparison is deterministic and does not rerun the model.

Single-product and competitor reports are available in English or Simplified Chinese. Each report is a standalone HTML file with embedded CSS, escaped dynamic content, no CDN, no external font, and no JavaScript dependency. It remains readable after Streamlit is closed and can be printed or saved as PDF in a browser.

## Privacy boundary

- The v1.0 default runs inference locally through Ollama.
- Starting the Dashboard, browsing Demo, changing language, loading snapshots, comparing products, and exporting reports do not trigger hidden model calls.
- Amazon Quick Scan is the only workflow that accesses Amazon.
- User data and generated artifacts remain local by default; there is no automatic cloud synchronization.
- Secrets, cookies, browser profiles, the Python virtual environment, and model files are excluded from the release portfolio package.

## Quality evidence

- Product Release Acceptance: **PASS**
- Automated tests: **1077 / 1077 passed**
- Open release issues: **0 BLOCKER / 0 MAJOR**
- Frozen Blind V2 Aspect + Sentiment F1: **85.66%** on 150 cases
- Frozen Blind V2 Strict Exact Case Accuracy: **66.67%**

The evaluation figures are specific to the frozen dataset and v1.0 configuration. They should not be presented as universal accuracy.

## Product boundaries

v1.0 is a local decision-support product, not an automated replacement for product or ecommerce professionals. It does not guarantee Amazon access, exhaustive review coverage, statistical representativeness, causal findings, or a universal model-quality level. Comparison supports two products, and the fixed taxonomy may map category-specific attributes to Other.
