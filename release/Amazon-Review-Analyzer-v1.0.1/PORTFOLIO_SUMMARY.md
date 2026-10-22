# Amazon Review Analyzer v1.0.1 — Portfolio Summary

## Product

Amazon Review Analyzer is a local-first product review intelligence application for macOS. It turns unstructured reviews into traceable product attributes, sentiment, source Evidence, customer language, and decision-support views.

## What it demonstrates

- local AI review analysis through Ollama and `qwen3:8b`
- Aspect-level sentiment rather than one label for an entire review
- strict Evidence grounding back to the original review
- review-level aggregation, keywords, and deterministic Business Intelligence
- bilingual English / Simplified Chinese Dashboard
- durable Saved Analyses with compatibility and SHA-256 checks
- transparent two-product competitor comparison
- standalone English and Chinese HTML business reports
- compliant browser-assisted Amazon Quick Scan with Paste and Upload fallbacks
- checkpoint, Resume, retry-failed, and stable 1-worker execution
- macOS `.app` launcher, First Run, Environment Health, Limited Mode, diagnostics, setup, install, and safe stop helpers

## Architecture

```text
Input → Validation → Cleaner → Local AI
      → Aspect + Sentiment + Evidence
      → Aggregation + Customer Language
      → Business Intelligence + Dashboard
      → Saved Analysis + Comparison + Reports
```

The system deliberately separates model interpretation from deterministic aggregation and presentation. Original comments and Evidence remain visible so a stakeholder can inspect why a conclusion appears.

## Frozen evaluation reference

Blind V2 is a frozen 150-case evaluation with:

- Aspect + Sentiment F1: 85.66%
- Aspect-only F1: 87.11%
- Sentiment Accuracy: 98.33%
- Evidence Validity: 100%
- Evidence Match: 100%

These metrics apply only to the frozen Blind V2 evaluation set and do not represent universal accuracy. v1.0.1 did not rerun or tune against the frozen evaluation.

## Distribution statement

v1.0.1 is a Stable Local Release and a portfolio/local-distribution package. The macOS `.app` is a lightweight launcher that requires the configured source project and local environment. It is not a fully standalone AI binary and does not bundle Python, `.venv`, Ollama, or the model. It is currently unsigned and not notarized.

## Honest boundaries

The product does not claim to scrape all Amazon reviews, provide an official Amazon API integration, achieve universal or 100% accuracy, make statistically significant causal claims, or choose an overall product winner. Amazon access depends on the live website and current user-visible session, and the app never bypasses CAPTCHA, login, anti-bot controls, or session restrictions.
