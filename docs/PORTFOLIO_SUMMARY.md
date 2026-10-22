# Amazon Review Analyzer — Portfolio Summary

## Short (~50 words)

Built a local AI review-intelligence product that converts Amazon customer feedback into aspect-level sentiment, verbatim evidence, business insights, competitor comparisons, and standalone reports. The Python/Streamlit application uses Ollama with qwen3:8b, supports bilingual workflows and resilient batch processing, and achieved 85.66% Aspect + Sentiment F1 on a frozen 150-case evaluation set.

## Medium (~120 words)

Designed and built Amazon Review Analyzer, a local-first Python and Streamlit product that turns unstructured e-commerce reviews into structured aspect, sentiment, evidence, keyword, and business-intelligence outputs. The system uses qwen3:8b through Ollama for semantic interpretation, then applies deterministic Python logic for aggregation, customer pain points, selling points, opportunity ranking, and two-product comparison. I added evidence validation, checkpoint/resume, targeted retry, bilingual UI, saved analyses with integrity checks, standalone HTML reports, and a lightweight macOS launcher. On the frozen 150-case Blind V2 set, it reached 85.66% Aspect + Sentiment F1, 98.33% sentiment accuracy, and 100% evidence validity/match. Metrics are evaluation-set specific, and Amazon access limitations are handled transparently through visible-browser import plus Paste/CSV fallbacks.

## Detailed (~250 words)

Amazon Review Analyzer is a local AI-powered e-commerce analytics product I developed to turn unstructured customer reviews into traceable product and competitor intelligence. Users can explore a frozen Demo Dataset, paste reviews, upload CSV data, or scan reviews available in a visible Amazon browser session. The system preserves original text, extracts multiple product aspects with aspect-level sentiment, validates verbatim evidence, and exposes every conclusion back to its source review.

The architecture intentionally separates semantic reasoning from deterministic analytics. qwen3:8b runs locally through Ollama for review interpretation; Python modules calculate review share, sentiment distribution, selling points, pain points, opportunity ranking, rating insights, keywords, and product-versus-shopping-experience signals. Completed results can be saved as integrity-checked local snapshots, compared across two products, and exported as standalone English or Simplified Chinese HTML reports.

I treated evaluation and reliability as product requirements. The batch runtime supports per-review checkpoints, resume, failed-review retry, and a single bounded retry for transient provider failures. A frozen evaluation process separated optimization data from final measurement. On the 150-case Blind V2 set, the system achieved 85.66% Aspect + Sentiment F1, 87.11% Aspect-only F1, 98.33% sentiment accuracy, and 100% evidence validity and match; these results apply only to that dataset.

The project also includes bilingual onboarding, Environment Health, Limited Mode, diagnostics, a five-tab Streamlit dashboard, and a lightweight macOS launcher. AI-assisted coding tools supported implementation, while product requirements, evaluation criteria, business rules, safety boundaries, and iteration decisions were explicitly designed throughout the project. It demonstrates practical capability across local AI, e-commerce, analytics engineering, evaluation, and productization.
