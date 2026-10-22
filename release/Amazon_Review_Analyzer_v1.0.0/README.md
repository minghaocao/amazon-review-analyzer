# Amazon Review Analyzer v1.0.0

**Local AI Review Intelligence & Competitive Analysis**

Turn customer reviews into structured product insights, evidence-backed business intelligence, and competitor comparisons using a local LLM.

> 基于本地 AI 的商品评论分析、商业洞察与竞品比较工具。

**Release:** Stable · **Validation:** PASS · **Automated tests:** 1077 / 1077 · **Release issues:** 0 BLOCKER / 0 MAJOR

This release/portfolio package accompanies the application source project. Run installation and startup commands from the source project root, where `requirements.txt`, `.env.example`, `app.py`, and the launcher configuration are available; this ZIP is not a standalone source distribution.

## What this product does

Amazon Review Analyzer turns unstructured review text into an auditable decision-support workflow. It identifies product and shopping-experience aspects, evaluates sentiment at the aspect level, keeps the exact customer wording that supports each result, and transforms those results into dashboards, saved analyses, two-product comparisons, and portable reports.

The default AI stack is local **Ollama + `qwen3:8b`**. The built-in Demo Dataset can be explored without Ollama or network access; the model is used only when the user explicitly analyzes new reviews.

## v1.0 at a glance

| Product scope | v1.0 |
| --- | ---: |
| Input modes | 4 |
| Canonical aspects | 15 |
| Dashboard views | 5 |
| Interface languages | 2 |
| Report types | 2 |
| Frozen Blind V2 cases | 150 |
| Automated tests passed | 1077 |
| BLOCKER / MAJOR issues | 0 / 0 |

## Feature overview

| Capability | Status |
| --- | :---: |
| Amazon Quick Scan | ✅ |
| CSV Upload | ✅ |
| Paste Reviews | ✅ |
| Aspect Analysis | ✅ |
| Sentiment Analysis | ✅ |
| Evidence Extraction | ✅ |
| Business Intelligence | ✅ |
| Saved Analyses | ✅ |
| Competitor Comparison | ✅ |
| English / 简体中文 | ✅ |
| Standalone HTML Reports | ✅ |
| Checkpoint / Resume | ✅ |
| Local LLM | ✅ |

## From customer language to business evidence

```text
Amazon Quick Scan / CSV / Paste / Demo
                    ↓
          Validation and cleaning
                    ↓
      Local aspect-based AI analysis
                    ↓
      Aspect + Sentiment + Evidence
                    ↓
 Aggregation + Keywords + Business Intelligence
                    ↓
 Dashboard → Saved Analysis → A/B Comparison → Reports
```

The model does not assign only one sentiment to an entire review. A single review can say that picture quality is excellent while the remote is disappointing. Each supported result keeps an Evidence fragment that must occur as a continuous substring of the original review. Invalid Evidence is rejected rather than silently rewritten.

## Product experience

- **Overview** — KPIs, coverage, executive summary, customer concerns, selling points, and pain points.
- **Product Insights** — opportunity ranking, priority matrix, rating insights, hidden complaints, and deterministic recommendations.
- **Customer Language** — commercially useful words and phrases, including bigrams and trigrams.
- **Aspect Deep Dive** — aspect-level review share, sentiment rates, sample evidence level, keywords, and source Evidence.
- **Review Explorer** — trace every result back to the original review, summary, aspect, sentiment, and Evidence.

The UI can switch between English and Simplified Chinese without rerunning analysis. Original reviews, Evidence, and customer phrases remain in their original language.

### Product preview

![Overview dashboard with KPIs and evidence-backed summary](screenshots/01_overview.png)

![Product Insights with customer-priority analysis](screenshots/02_product_insights.png)

More product views are available in the [`screenshots/`](screenshots/) folder.

## Inputs

| Mode | Best for | Important boundary |
| --- | --- | --- |
| Demo Dataset | Fast, offline product tour | Frozen sample; no live AI required |
| Paste Reviews | A quick real-model trial | One non-empty review per line |
| Upload Dataset | Controlled batch datasets | CSV up to 10 MB; review content required |
| Amazon Quick Scan | Reviews visible in a normal Amazon browser session | Access, login, verification, and available-review count depend on Amazon |

Amazon Quick Scan does not bypass CAPTCHA, login, Robot Check, Access Denied, or other website controls. If Amazon limits access, users can continue with Paste Reviews or Upload Dataset.

## Evidence-backed business intelligence

The system distinguishes product aspects from shopping experience:

- **Product:** Picture Quality, Sound Quality, Price / Value, Remote Control, System Performance, Ease of Use, Design, Build Quality, Reliability, Connectivity, and Setup / Installation.
- **Shopping experience:** Delivery, Packaging, and Customer Service.
- **Other:** retained for traceability but excluded from the primary commercial rankings by default.

Selling points, pain points, opportunities, and competitor observations use transparent deterministic rules over the analyzed review sample. They are directional decision-support signals—not statistical significance tests, market-share estimates, or automatic business decisions.

## Saved analysis and comparison

A completed analysis can be stored in the local Analysis Library as a versioned snapshot with integrity metadata and SHA-256 validation. A saved snapshot can be reopened after the app restarts without rerunning the model.

Competitor Comparison reads two compatible saved analyses and compares the same 15-aspect schema. It surfaces advantages, weaknesses, white-space opportunities, ratings, customer language, and source Evidence. It does not call the model or invent an overall winner.

## Exportable reports

v1.0 exports two report types in English or Simplified Chinese:

1. Single Product Report
2. Competitor Comparison Report

Reports are standalone HTML documents with embedded CSS. They do not depend on Streamlit, localhost, a CDN, external fonts, or JavaScript, and can be printed or saved as PDF from a browser.

## Evaluation context

The frozen 150-case Blind V2 evaluation produced:

- Aspect + Sentiment F1: **85.66%**
- Aspect-only F1: **87.11%**
- Sentiment Accuracy: **98.33%**
- Evidence Validity: **100%**
- Evidence Match: **100%**
- Strict Exact Case Accuracy: **66.67%**

These are results for one frozen evaluation set and the documented v1.0 model configuration. They are not a universal accuracy claim, a confidence interval, or a guarantee for every product category, language, writing style, or real Amazon dataset. See [Release Notes](RELEASE_NOTES.md) for the full release context.

## Runtime and privacy

- Default inference runs locally through Ollama with `qwen3:8b`.
- Stable production mode uses one worker, row-level checkpoints, Resume, and Retry Failed Reviews.
- Automatic retry is limited to one retry for recognized transient provider/connection failures; quality and schema errors are not retried automatically.
- Original reviews, current analyses, saved snapshots, comparisons, and reports remain in the local project workspace by default.
- Amazon Quick Scan is the only input mode that visits Amazon. Optional provider architecture does not change the v1.0 local default.

The release package intentionally excludes `.env`, credentials, cookies, browser profiles, `.venv`, Ollama model files, caches, and private runtime data.

## Requirements

The accepted v1.0 desktop environment is macOS with:

- Python 3.11 or later
- Ollama
- `qwen3:8b` installed separately through Ollama
- Project dependencies from `requirements.txt`
- Playwright Chromium only when Amazon Quick Scan is required

The model and virtual environment are not bundled with this portfolio package.

## Start here

- [中文快速开始](QUICK_START_CN.md)
- [English Quick Start](QUICK_START_EN.md)

For the fastest no-risk tour, launch the app and keep the default **Demo Dataset** selected. To try real local inference, use a small Paste Reviews sample after confirming Ollama and `qwen3:8b` are ready.

## Documentation

- [中文完整系统说明（Markdown）](docs/Amazon_Review_Analyzer_v1.0_User_Guide_CN.md)
- [中文完整系统说明（Standalone HTML）](docs/Amazon_Review_Analyzer_v1.0_User_Guide_CN.html)
- [Release Notes](RELEASE_NOTES.md)
- [Changelog](CHANGELOG.md)
- [System Overview](SYSTEM_OVERVIEW.md)
- [Portfolio Summary](PORTFOLIO_SUMMARY.md)

Product screenshots are stored in [`screenshots/`](screenshots/), safe demonstration data in [`sample_data/`](sample_data/), and pre-rendered report examples in [`sample_reports/`](sample_reports/).

## Known boundaries

- Amazon can require login or human verification and does not guarantee that all reviews are accessible.
- Local inference speed depends on hardware, model state, review length, and background load.
- Small samples are early signals and should not be treated as statistically representative.
- v1.0 compares exactly two saved products at a time.
- The taxonomy contains 15 canonical categories; category-specific details may map to Other.
- HTML can be printed to PDF, but v1.0 does not generate a native PDF file.
- The product is local-first; it does not provide cloud accounts, collaborative workspaces, or a shared database.

## License status

No software license is included in this release package. Do not assume that public viewing grants permission to copy, redistribute, or reuse the source code.

---

Amazon Review Analyzer v1.0.0 is designed as a transparent review-intelligence product: conclusions stay connected to customer language, sample limitations remain visible, and the human user remains responsible for business decisions.
