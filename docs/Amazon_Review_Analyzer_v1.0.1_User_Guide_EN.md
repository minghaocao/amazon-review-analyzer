# Amazon Review Analyzer v1.0.1 — User Guide

**Status:** Stable Local Release  
**Release type:** Maintenance / UX / Distribution Release

Amazon Review Analyzer is a local-first macOS application for evidence-grounded product review analysis. It extracts Aspect-level sentiment and source Evidence, aggregates customer concerns, highlights customer language, saves completed analyses, compares two products, and exports standalone HTML reports.

## 1. Recommended Local Setup

1. Install Ollama from its official source.
2. Confirm that `qwen3:8b` is available locally. If needed, explicitly run `ollama pull qwen3:8b`.
3. Open the source project in Finder and run `setup_amazon_review_analyzer.command`.
4. Let Setup check Python 3.11+, create a missing `.venv`, install project dependencies, and inspect the local AI environment.
5. Open `Amazon Review Analyzer.app` from the project or install it with the supplied helper.
6. The app starts the local Streamlit interface at `http://localhost:8501`.
7. Begin with Demo Dataset.

Setup does not silently install Ollama, download the model, contact Amazon, run review inference, delete an existing environment, or overwrite `.env`.

## 2. Desktop experience

The first successful launch presents bilingual onboarding and the current Environment Health state. If Ollama or `qwen3:8b` is unavailable, **Limited Mode** keeps Demo, Saved Analyses, existing comparisons, reports, and documentation available while blocking only actions that require new inference.

The launcher reuses an existing instance of this project instead of creating a second Streamlit listener. Use `Stop Amazon Review Analyzer.command` to stop only the validated project Dashboard; it does not stop Ollama or unrelated Python processes.

## 3. Input modes

- **Demo Dataset:** frozen local demonstration data; no model or network required.
- **Paste Reviews:** one review per line with preview, validation, and explicit analysis start.
- **Upload Dataset:** CSV import with column mapping; review content is required and other metadata is optional.
- **Amazon Quick Scan:** browser-assisted import of only the reviews available to the ordinary visible browser session.

For CSV, use `content` as the required text field. Recommended fields are `review_id` and `rating`; `title`, `date`, and `verified_purchase` are optional.

## 4. Analysis and runtime

New analysis uses local Ollama with `qwen3:8b`. Each completed review preserves its original content and produces a summary plus Aspect entries containing the canonical Aspect, sentiment, and source Evidence. Evidence must be a continuous substring of the review.

Stable runtime uses 1 worker. Checkpoints support Resume without reprocessing `success` or `skipped` reviews. Retry Failed Reviews targets only failed rows. Automatic retry is limited to one retry for explicit transient provider/connection failures; schema, Evidence, content, and user-input errors are not automatically retried.

## 5. Dashboard and outputs

The Dashboard provides Overview, Product Insights, Customer Language, Aspect Deep Dive, and Review Explorer. Completed analyses can be saved locally with integrity metadata, compared in a two-product A/B view, and exported as standalone English or Chinese HTML reports.

Original reviews, Evidence, keywords, and customer phrases remain in their source language. Changing the UI language does not rerun analysis.

## 6. Privacy

- The default configuration requires no OpenAI API key.
- Reviews, checkpoints, outputs, saved analyses, comparisons, and reports remain in the local project workspace by default.
- Starting the app, viewing Demo, running Health Check, switching language, or opening completed content does not call OpenAI or Amazon.
- Amazon Quick Scan is the only input mode that contacts Amazon.
- Browser cookies, credentials, and session data are not written into reports or Saved Analysis.

## 7. Amazon access limitations

Amazon may require login, request human verification, challenge browser sessions, provide partial reviews, or restrict review pages. Quick Scan cannot guarantee complete retrieval. It never bypasses CAPTCHA, login, anti-bot controls, or session restrictions. Use Paste Reviews or Upload Dataset when current website access is unavailable.

## 8. Local model limitations

New AI analysis requires a running Ollama service and local `qwen3:8b`. Runtime depends on Mac hardware, review length, and cold-start state. Small datasets may not meet ranking thresholds; this is an evidence-safety behavior, not an application error.

## 9. App packaging limitations

The macOS `.app` is a lightweight launcher for an existing local project installation. It does not contain Python, `.venv`, Ollama, `qwen3:8b`, or the full backend, and it is not a fully self-contained binary. The current bundle is not code signed or notarized for public distribution. On another Mac, use Finder **Right Click → Open** if macOS requests confirmation; do not weaken platform security.

## 10. Evaluation context

Frozen Blind V2 contains 150 cases. Its frozen results are Aspect + Sentiment F1 85.66%, Aspect-only F1 87.11%, Sentiment Accuracy 98.33%, Evidence Validity 100%, and Evidence Match 100%.

These metrics apply only to the frozen Blind V2 evaluation set and do not represent universal accuracy. Frozen evaluations were not rerun or used for model changes in v1.0.1.

## 11. Troubleshooting

- **Missing `.venv` or dependencies:** rerun the setup helper from the source project.
- **Ollama unavailable:** open Ollama and refresh System Status; viewing features remain available in Limited Mode.
- **Model missing:** explicitly run `ollama pull qwen3:8b`, then refresh status.
- **Port 8501 in use:** open `http://localhost:8501` first; if another app owns the port, close it normally.
- **Amazon access limited:** complete any requested user verification in the visible browser, or switch to Paste/Upload.
- **Interrupted analysis:** restore the local AI environment, reopen the same Current Analysis, and use Resume.
- **Unsigned app blocked:** in Finder, use Right Click → Open once and accept the normal macOS prompt.

## 12. Release scope

v1.0.1 formalizes First Run, Environment Health, Limited Mode, Diagnostics, the macOS launcher, install helper, safe stop, single-instance behavior, and bilingual startup guidance. It does not change the model, Prompt, taxonomy, sentiment logic, Evidence validation, Aggregator, Keyword Analyzer, Amazon strategy, Business Intelligence, comparison semantics, report semantics, or frozen evaluation results.
