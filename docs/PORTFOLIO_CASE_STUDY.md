# Amazon Review Analyzer — Product and Engineering Case Study

## 1. Project Background

Amazon Review Analyzer began as a focused question: can a local AI system turn individual product reviews into structured insights without losing the original customer evidence? The project evolved from a CSV-based prototype into a bilingual macOS product with traceable review analysis, deterministic business intelligence, saved analyses, competitor comparison, and standalone reporting.

Version 1.0.1 is a stable local portfolio release. It is an independent research project and is not affiliated with or endorsed by Amazon.

## 2. Business Problem

Product reviews are valuable but operationally difficult to use. Important signals are distributed across hundreds of comments, mixed with delivery feedback, contradictory opinions, vague language, and repeated mentions. Star ratings alone cannot explain why customers are satisfied or dissatisfied.

A product manager, seller, or analyst typically wants to know:

- which product attributes customers discuss most;
- which attributes drive praise or frustration;
- how often an issue appears across different reviewers;
- which customer phrases describe the problem;
- what evidence supports each conclusion;
- where one product is stronger or weaker than another.

Manual reading can answer these questions for a small sample, but it is slow, inconsistent, and hard to reproduce.

## 3. Product Goal

The goal was to build a transparent review-intelligence workflow rather than a generic summarizer. The product needed to convert raw reviews into structured aspect, sentiment, and evidence records, then aggregate those records into useful business signals.

Three principles guided the product:

1. **Evidence before conclusions.** Every extracted aspect should point back to verbatim review text.
2. **Semantic reasoning plus deterministic analytics.** Use a language model for interpretation, then use explicit Python rules for counts, rates, ranking, and reporting.
3. **Local-first operation.** Keep the default analysis path private, inspectable, and independent of per-request cloud API costs.

## 4. Initial Prototype

The first usable version deliberately avoided live Amazon access. It accepted manually entered reviews or CSV files, validated the schema, preserved the original content, and cleaned text without removing punctuation, numbers, capitalization, or negation.

That sequencing was intentional. The review-analysis problem could be measured and improved independently from website-access problems. Once the structured analysis pipeline was stable, additional input modes and product surfaces were added around it.

## 5. Key Engineering Challenges

The project presented several distinct challenges:

- extracting multiple aspect-sentiment pairs from one review;
- normalizing varied customer language into a stable taxonomy;
- preserving exact evidence rather than accepting model paraphrases;
- recovering from individual provider failures without losing an entire batch;
- avoiding duplicate customer counts when one review repeats an aspect;
- separating product quality from delivery, packaging, and customer service;
- communicating uncertainty when sample sizes are small;
- maintaining compatibility across the dashboard, saved analyses, comparisons, and reports;
- making a Python/Streamlit project feel reliable in a desktop interview demo.

The final architecture addresses these as separate layers rather than one large prompt or script.

## 6. Amazon Access-Control Challenge

Live Amazon access is inherently variable. Amazon can require login, show human verification, restrict visible review pages, or change page structure. An early direct-request approach exposed the difference between network/TLS failures and deliberate website access controls.

The resulting product strategy is conservative:

- Amazon Quick Scan uses an ordinary visible browser session;
- users complete any required login or verification themselves;
- the application never bypasses CAPTCHA or Amazon access controls;
- only reviews accessible to the current session are imported;
- Paste Reviews and Upload Dataset are first-class fallback methods;
- the Demo Dataset provides a stable offline experience.

This constraint is presented honestly as a limitation rather than hidden behind a “scrape all reviews” claim.

## 7. Local LLM Decision

The default provider is Ollama with `qwen3:8b`. Local inference was selected for four practical reasons:

- review data remains on the user’s machine by default;
- no per-request API charge is required;
- model and prompt versions can be frozen for repeatable evaluation;
- the product remains useful offline once the environment is installed.

The trade-off is speed and environment complexity. New analysis requires Ollama, the local model, and a suitable Mac. The application therefore includes Environment Health checks, Limited Mode, diagnostics, and a frozen Demo experience that works without model inference.

## 8. Review Analysis Pipeline

The production flow is:

```text
Input
→ schema validation
→ conservative text cleaning
→ local semantic analysis
→ aspect normalization
→ evidence validation
→ per-review persistence
→ deterministic aggregation
→ keyword and phrase analysis
→ business intelligence
→ dashboard, saved analysis, comparison, and reports
```

The model returns a summary and aspect mentions. Python post-processing validates schema and evidence, derives positive and negative points, preserves raw mentions, and creates review-level unique aspects for aggregation. The batch layer handles checkpoints, resume, skipped rows, failed rows, and a single bounded retry only for transient provider connection errors.

## 9. Evidence-Grounded Design

Evidence is a core data contract, not decorative text. An aspect’s evidence must be a continuous substring of the original review. If one aspect has invalid evidence, valid aspects from the same review are preserved. The invalid item receives one focused evidence-repair attempt; if it still fails, only that item is discarded and a warning is recorded.

This design reduces a common failure mode in LLM applications: plausible conclusions that cannot be traced to source material. The Review Explorer and Aspect Deep Dive expose the original review, summary, normalized aspect, sentiment, and evidence together.

## 10. Evaluation Methodology

Evaluation was implemented as a standalone module so model or prompt changes could be measured against human-labeled answers. Development sets and final evaluation sets were kept separate. Frozen sets were hashed and treated as protected assets after use.

Metrics include:

- aspect + sentiment precision, recall, and F1;
- aspect-only precision, recall, and F1;
- sentiment accuracy and confusion summaries;
- evidence validity and evidence match;
- strict exact-case accuracy.

Strict exact matching considers aspect identity, sentiment, mention count, and evidence while ignoring only mention order. This is a demanding measure and exposes errors that aggregate metrics can hide.

## 11. Blind Evaluation Results

On the frozen 150-case Blind V2 evaluation set:

| Metric | Result |
| --- | ---: |
| Aspect + Sentiment F1 | 85.66% |
| Aspect-only F1 | 87.11% |
| Sentiment Accuracy | 98.33% |
| Evidence Validity | 100% |
| Evidence Match | 100% |

These results apply only to that frozen dataset. They are not universal accuracy claims, and they do not guarantee the same performance for every category, language, or review style. The separation between optimization and blind evaluation was used to reduce leakage and make the final numbers more meaningful.

## 12. Runtime Optimization

Performance work focused on practical stability rather than uncontrolled parallelism. Model output was reduced to the fields that require semantic reasoning; repeated positive and negative point lists are derived from validated aspects in Python. Ollama keep-alive avoids needless model unloading, and the context window remains intentionally bounded for single-review analysis.

A frozen 20-review benchmark compared serial processing with two concurrent workers on the current 16 GiB Mac environment. Two workers were slower and produced worse latency, so the stable product configuration uses one worker. The lesson was hardware-specific: concurrency is not automatically throughput.

## 13. Business Intelligence Layer

The business layer is deterministic and operates only on completed structured outputs. It calculates review share, positive and negative review counts, sentiment distribution, selling points, pain points, opportunity ranking, priority matrix, rating insights, and hidden complaints.

Product aspects are separated from shopping-experience aspects. Delivery, Packaging, and Customer Service can appear in customer concerns and shopping-experience insights but do not become product selling points. Formal rankings require a minimum sample size and compare positive versus negative review counts. The UI preserves raw rates and evidence levels instead of inventing a composite “truth score.”

These signals are directional aids, not statistical significance claims.

## 14. Competitor Comparison

The comparison workflow uses two previously analyzed, integrity-checked snapshots. It aligns a fixed aspect taxonomy and produces head-to-head coverage, sentiment differences, advantages, weaknesses, white-space opportunities, rating context, customer language, and side-by-side evidence.

Comparison never reruns the model. It also avoids declaring a universal winner when samples are small or uneven. Missing evidence remains “No evidence” rather than being converted into a misleading zero-percent score.

## 15. Localization

The dashboard and reports support English and Simplified Chinese. Localization is intentionally a presentation boundary: canonical aspect names, sentiment values, CSV columns, and JSON keys remain stable in English, while UI labels and deterministic prose are translated.

Original reviews, evidence, keywords, and customer phrases are never silently translated. This preserves the customer’s language and prevents localization from altering the analytical source.

## 16. Saved Analysis Architecture

Completed analyses can be saved as local snapshots with version metadata, schema identifiers, analysis-candidate identifiers, and SHA-256 hashes for critical files. Snapshots survive Streamlit restarts and can be opened, compared, or used for report generation without new inference.

Integrity and compatibility checks fail safely. The system does not ignore hash mismatches or silently combine incompatible analysis versions. This turns a temporary dashboard run into a reusable analytical asset while keeping persistence local and inspectable.

## 17. Reporting System

Single-product and competitor-comparison reports are exported as standalone English or Chinese HTML files. CSS is embedded, dynamic content is escaped, and the reports do not depend on Streamlit, localhost, a CDN, or external JavaScript after generation.

Reports include source context, sample-size warnings, business insights, and traceable evidence. They can be opened directly or printed to PDF from the browser. Generation reads existing outputs only and does not call the model or Amazon.

## 18. Productization

The project moved beyond analysis scripts into a local product experience:

- a five-tab Streamlit dashboard;
- bilingual first-run onboarding;
- deterministic Environment Health and Advanced Diagnostics;
- Limited Mode when Ollama or the model is unavailable;
- a lightweight macOS `.app` launcher;
- install and safe-stop helpers;
- single-instance handling and startup recovery;
- curated screenshots, sample reports, release packages, and security scans.

The `.app` is accurately described as a lightweight launcher. It is unsigned, not notarized, and depends on the local project, Python environment, Ollama, and `qwen3:8b`.

## 19. Key Lessons

Several lessons shaped the final product:

- Separate semantic interpretation from deterministic calculation.
- Design evidence validation as a hard contract early.
- Treat failure recovery and observability as product features.
- Keep live website access optional and isolate it from core analysis.
- Measure concurrency on the target hardware instead of assuming it helps.
- Freeze final evaluation assets before drawing portfolio conclusions.
- Show sample size and uncertainty alongside every business ranking.
- Preserve compatibility when productizing a mature analysis pipeline.

AI-assisted coding tools were used as development tooling throughout the project. Product requirements, evaluation criteria, business rules, safety boundaries, data contracts, and iteration decisions were designed and reviewed across the development process. The project does not claim that every line was written manually; it demonstrates how AI-assisted implementation can be directed by explicit product and engineering judgment.

## 20. Future Roadmap

Potential next steps include broader category-specific taxonomies, more multilingual evaluation, controlled multi-product comparison, optional database-backed project management, additional model benchmarking, and signed/notarized packaging. Any future work should preserve the current evidence contract, frozen evaluation history, and clear boundary between semantic reasoning and deterministic analytics.

The immediate v1.0.1 goal is complete: a stable local portfolio release that communicates both its capabilities and its limitations.
