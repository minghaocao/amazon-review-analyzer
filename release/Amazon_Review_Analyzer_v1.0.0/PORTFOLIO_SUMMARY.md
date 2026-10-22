# Amazon Review Analyzer v1.0.0 — Portfolio Summary

**Local AI Review Intelligence & Competitive Analysis**

Amazon Review Analyzer is a local-first desktop product that turns unstructured customer reviews into traceable product intelligence. It combines aspect-based sentiment analysis, exact customer Evidence, deterministic business metrics, saved analytical snapshots, and two-product comparison in a bilingual Streamlit experience.

## The problem

Product and ecommerce teams often have thousands of reviews but limited time to read them. A star average says little about *why* customers are satisfied or frustrated. Generic sentiment analysis also loses important nuance: the same customer may praise picture quality while criticizing the remote, software performance, or delivery.

The harder product problem is not merely summarization. Useful review intelligence must answer:

- Which product attributes matter most to customers?
- Which strengths appear consistently enough to support positioning?
- Which pain points deserve investigation?
- What language do customers actually use?
- Can every conclusion be traced back to source reviews?
- How does the same evidence structure compare across two products?

## The solution

I built a complete local analysis workflow that converts each review into structured **Aspect + Sentiment + Evidence** records. The model handles semantic interpretation; Python then validates Evidence, normalizes aspects, aggregates at review level, calculates transparent business signals, and presents the result through dashboards and portable reports.

```text
Reviews
   ↓
Validation and cleaning
   ↓
Local qwen3:8b analysis
   ↓
Aspect + Sentiment + exact Evidence
   ↓
Deterministic aggregation and business intelligence
   ↓
Dashboard → Saved Analysis → Competitor Comparison → Reports
```

## v1.0 at a glance

| Measure | Delivered |
| --- | ---: |
| Input modes | 4 |
| Canonical aspects | 15 |
| Dashboard views | 5 |
| UI languages | 2 |
| Report types | 2 |
| Frozen Blind V2 cases | 150 |
| Automated tests | 1077 / 1077 passed |
| Release gate | PASS |
| BLOCKER / MAJOR issues | 0 / 0 |

## What I built

### Review intelligence

- Aspect-Based Sentiment Analysis rather than one label for an entire review.
- A fixed 15-aspect taxonomy spanning product quality, usability, and shopping experience.
- Evidence validation that requires each supporting phrase to be a continuous substring of the original review.
- Review-level deduplication so repeated mentions by one customer do not appear as multiple customers, while the original mention count and Evidence remain available.
- Commercially useful keyword, bigram, and trigram extraction with source references.

### Business intelligence

- Customer concerns, product selling points, product pain points, and shopping-experience insights.
- Opportunity Ranking and a Customer Priority Matrix combining review share with sentiment direction.
- Rating-aware views, including complaints inside 4–5 star reviews and strengths inside 1–2 star reviews.
- Deterministic recommendations that explain the sample evidence behind each signal.
- Clear separation between product issues and Delivery, Packaging, or Customer Service.

### Product experience

- Four entry points: offline Demo, Paste Reviews, mapped CSV Upload, and browser-assisted Amazon Quick Scan.
- Five exploration views: Overview, Product Insights, Customer Language, Aspect Deep Dive, and Review Explorer.
- English and Simplified Chinese display without rerunning analysis or translating customer Evidence.
- Persistent Saved Analyses with stable IDs, version metadata, and SHA-256 integrity validation.
- Deterministic Product A vs Product B comparison with advantages, weaknesses, white-space opportunities, rating, language, and Evidence.
- Standalone English and Chinese HTML reports for both single-product and competitor results.
- A macOS one-click launcher and Demo-first behavior for reliable interviews and product demonstrations.

## Reliability engineering

Local LLM applications need more than a prompt. The production runtime uses one worker, checkpoints after every row, resumes compatible incomplete runs, retries only failed rows on request, and performs at most one automatic retry for recognized transient provider connection failures. One bad review does not stop a batch.

Completed outputs are isolated by source, and saving an analysis creates a persistent snapshot rather than a browser-only session object. Snapshot compatibility and file hashes are checked before a saved result is loaded, compared, or exported. Dynamic report content is HTML escaped, and reports are designed to contain no local absolute paths or hidden network dependencies.

The accepted release passed **1077 / 1077 automated tests**, with **0 BLOCKER** and **0 MAJOR** release issues.

## Evaluation

On the frozen 150-case Blind V2 set, the v1.0 `qwen3:8b` configuration achieved:

- **85.66% Aspect + Sentiment F1**
- **87.11% Aspect-only F1**
- **98.33% Sentiment Accuracy**
- **100% Evidence Validity** and **100% Evidence Match** under the evaluation protocol
- **66.67% Strict Exact Case Accuracy**

This is a frozen evaluation result, not a universal accuracy claim. It does not guarantee performance on every product category, language, review style, or live Amazon dataset. Strict Exact requires the case-level Aspect, sentiment, mention count, and Evidence to match the Gold definition, making it materially stricter than component metrics.

A separate frozen 20-review throughput benchmark selected **Stable mode / 1 worker**. It completed 20/20 reviews with a 12.14-second median per-review latency and 4.07 reviews per minute in the accepted environment. The benchmark measures runtime behavior rather than accuracy, and real speed depends on hardware and workload.

## Business value

The product helps an operator or product team:

- discover recurring strengths without losing source context;
- prioritize customer pain points by both prevalence and sentiment direction;
- identify hidden complaints even in highly rated reviews;
- preserve the exact customer phrases useful for listing copy and product language;
- compare two products using a consistent taxonomy and visible sample sizes;
- create a portable evidence-backed report for discussion or interview presentation.

It is decision support, not an automated replacement for the operator. Sample quality, representativeness, competitor comparability, and final business action remain human responsibilities.

## Technical highlights

- **Python and pandas** for data contracts, validation, aggregation, and export.
- **Streamlit and Plotly** for a bilingual interactive desktop experience.
- **Ollama + `qwen3:8b`** for local semantic analysis.
- **Pydantic-compatible structured output** and explicit Evidence validation.
- **Deterministic post-processing** for reproducible aggregation, scoring, comparison, and reports.
- **Checkpoint / Resume / Retry** for long-running local inference.
- **SHA-256 integrity controls** for frozen assets and saved analytical snapshots.
- **Standalone HTML** reporting with embedded CSS and escaped dynamic content.
- **Local-first privacy boundary** with no cloud account or automatic upload.

## Responsible scope

Amazon Quick Scan respects the visible browser session and does not bypass login, CAPTCHA, Robot Check, Access Denied, or other access controls. Review availability cannot be guaranteed, so CSV Upload and Paste Reviews remain first-class alternatives.

Business rankings are directional heuristics, not statistical significance tests or causal conclusions. Small samples are explicitly labeled, missing evidence is not represented as a zero score, and the system does not declare an unsupported overall competitor winner.

## Outcome

Amazon Review Analyzer v1.0.0 demonstrates an end-to-end product rather than an isolated model demo: real input handling, local inference, structured and validated outputs, failure recovery, business interpretation, persistent analysis artifacts, cross-product comparison, bilingual UX, report export, and release-level testing all operate under one coherent data contract.

For a product tour, begin with the [Release README](README.md). For the full operating reference, use the [中文系统说明](docs/Amazon_Review_Analyzer_v1.0_User_Guide_CN.md).
