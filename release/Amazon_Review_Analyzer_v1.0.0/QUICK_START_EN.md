# Amazon Review Analyzer v1.0.0 — Quick Start

Amazon Review Analyzer is a local-first macOS application that turns product reviews into traceable aspects, sentiment, source evidence, customer language, and business insights.

> The fastest way to explore the product is **Demo Dataset**. The demo works without Ollama, a downloaded model, or Amazon access.

This Quick Start accompanies the application source project. Run setup and startup commands from the source project root containing `requirements.txt`, `.env.example`, and `app.py`; the portfolio ZIP is not a standalone source distribution.

## 1. Requirements

- macOS
- Python 3.11 or later (v1.0 was accepted on Python 3.13.14)
- Ollama and model `qwen3:8b` (installed separately; required only for new analysis)
- Playwright Chromium for Amazon Quick Scan

The release does not bundle `.venv` or model files.

## 2. First-time setup

Install and start Ollama from the [official Ollama website](https://ollama.com/download/mac). Then open Terminal and move to the project root:

```bash
cd /path/to/amazon_review_analyzer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
test -f .env || cp .env.example .env
ollama pull qwen3:8b
python main.py --check-ollama
```

Do not replace an existing `.env`. The default local `Ollama + qwen3:8b` configuration does not require an OpenAI API key.

Install the browser runtime only if you plan to use **Amazon Quick Scan**:

```bash
source .venv/bin/activate
playwright install chromium
```

## 3. Start the application

### Option 1: Desktop launcher (recommended for daily use)

After first-time setup, double-click this file on the Desktop:

```text
Amazon Review Analyzer.command
```

The launcher enters its configured project directory, activates `.venv`, checks Ollama and the model, avoids duplicate Streamlit instances, and opens the Dashboard. If Ollama is not running, it makes a limited attempt to open the application.

The Demo Dashboard still opens when Ollama or the model is unavailable; only new AI analysis is disabled. Press `Ctrl+C` in the launcher Terminal window to stop that Dashboard process. This does not delete data or stop Ollama.

> After moving the project, update `PROJECT_DIR` in the launcher. If macOS blocks the file the first time, right-click it in Finder and choose **Open**.

### Option 2: Start from Terminal

For normal day-to-day startup:

```bash
cd /path/to/amazon_review_analyzer
source .venv/bin/activate
streamlit run app.py
```

Streamlit normally opens [http://localhost:8501](http://localhost:8501). The application starts in Demo Dataset mode and does not automatically call the model or contact Amazon.

## 4. Choose an input

Select a source under **Dataset Mode** in the sidebar.

### Option A — Demo Dataset

The quickest and most reliable walkthrough. The frozen demo has 15 reviews, 14 successful analyses, and 93.3% coverage, and shows all five Dashboard views offline.

### Option B — Paste Reviews

Open **Analyze New Dataset → Paste Reviews**, place one review on each non-empty line, preview the input, and then start local AI analysis.

### Option C — Upload Dataset

Open **Analyze New Dataset → Upload Dataset** and upload a CSV of up to 10 MB. The application asks you to map available columns:

- Review content (`content`): required
- `review_id`, `rating`, `title`, `date`, and `verified_purchase`: optional or recommended

The importer creates stable IDs when Review ID is missing. Ratings should be 1–5; rating-dependent views are hidden when ratings are unavailable.

### Option D — Amazon Quick Scan

Open **Analyze New Dataset → Amazon Quick Scan** and enter a supported Amazon marketplace product URL:

1. Select **Open Amazon Product** to launch an ordinary visible browser.
2. If Amazon requests login or human verification, complete it yourself in that browser.
3. Select **Scan Available Reviews** and inspect the preview. This step does not call the model.
4. Select **Analyze Available Reviews** only after the preview looks correct.

Quick Scan imports only what the ordinary browser session can access. It cannot guarantee complete retrieval and does not bypass CAPTCHA, login, or other controls. Use **Paste Reviews** or **Upload Dataset** when access is limited.

## 5. After analysis

The five views are **Overview, Product Insights, Customer Language, Aspect Deep Dive, and Review Explorer**. Save completed results to the local **Analysis Library** to reopen them, export a standalone HTML report, or compare two saved products.

## 6. Troubleshooting

- **Python virtual environment was not found** — from the project root, recreate `.venv` and reinstall `requirements.txt`. The launcher intentionally does not create the environment.
- **Local AI model is unavailable** — start Ollama, run `ollama list` to confirm `qwen3:8b`, then run `python main.py --check-ollama`.
- **Port 8501 is already in use** — open `http://localhost:8501` and check whether this project is already running; the launcher reuses it.
- **Amazon browser will not open** — activate `.venv` and run `playwright install chromium`; use CSV or pasted input if it remains unavailable.
- **Amazon requests verification or limits access** — complete legitimate verification yourself. If access remains limited, use Upload / Paste rather than attempting to bypass website controls.

By default, analysis runs through local Ollama and data remains in the local project workspace. Amazon Quick Scan is the only input mode that actively connects to Amazon.
