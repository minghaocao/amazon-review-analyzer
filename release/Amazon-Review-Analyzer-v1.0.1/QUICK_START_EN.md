# Amazon Review Analyzer v1.0.1 — Quick Start

**Status: Stable Local Release**

Amazon Review Analyzer is a local-first macOS application that turns product reviews into traceable aspects, sentiment, source evidence, customer language, and business insights.

> The fastest way to explore the product is **Demo Dataset**. The demo does not require Ollama, a downloaded model, or Amazon access.

## Recommended Local Setup

**First-time setup:** follow this recommended sequence for the v1.0.1 local release.

1. Install Ollama from its official source.
2. Confirm that `qwen3:8b` exists locally. If it is missing, explicitly run `ollama pull qwen3:8b`.
3. Open the source project folder in Finder and double-click `setup_amazon_review_analyzer.command`. Setup checks Python, creates `.venv` when missing, installs `requirements.txt`, and inspects the local AI environment.
4. When the final status is **READY**, either open `dist/Amazon Review Analyzer.app` or use the install helper to copy it to Applications.
5. The app starts the local Streamlit UI and opens [http://localhost:8501](http://localhost:8501).
6. Begin with **Demo Dataset** before analyzing new reviews.

The development `.app` is unsigned and not notarized. If Gatekeeper blocks it the first time, right-click the app in Finder and choose **Open**, then confirm the macOS prompt. Do not remove quarantine attributes, disable Gatekeeper, or weaken macOS security settings. Setup may show the same normal right-click **Open** requirement for its `.command` file.

The `.app` is a lightweight launcher, not a self-contained binary. It requires the source project, a working project environment, and local dependencies. It does not contain Python, `.venv`, Ollama, `qwen3:8b`, or the full backend.

Setup does not:

- silently install Ollama;
- automatically download the large `qwen3:8b` model;
- contact Amazon;
- run review inference;
- delete an existing `.venv`; or
- replace an existing `.env`.

## Requirements

- macOS
- Python 3.11 or later
- Ollama and `qwen3:8b` for new review analysis
- Playwright Chromium for Amazon Quick Scan

If Ollama is not installed, use the [official Ollama website](https://ollama.com/download/mac). Setup checks system software but does not install it for you.

If Setup reports that the model is missing, explicitly run:

```bash
ollama pull qwen3:8b
```

The model requires several GB of local storage, so Setup never starts this download automatically.

## Check environment health

Activate the project environment and run:

```bash
source .venv/bin/activate
python main.py --health-check
```

For machine-readable output:

```bash
python main.py --health-check-json
```

Exit codes:

| Code | Meaning |
| ---: | --- |
| `0` | Ready; the app can launch and analyze new reviews |
| `1` | Action Required / Limited; the Dashboard works, but a component needs attention |
| `2` | Unavailable; the Python environment or a core runtime directory cannot be used safely |

Health Check inspects local files, packages, processes, the port, and lightweight Ollama metadata. It does not run model inference or contact Amazon.

## Recommended Launch

### Native macOS launcher (recommended)

Double-click:

```text
dist/Amazon Review Analyzer.app
```

The app is a small launcher wrapper. It finds the project, reuses the existing environment health check, starts the local Streamlit Dashboard in the background, waits until it is ready, and opens the default browser. It does not bundle Python, `.venv`, Ollama, or `qwen3:8b`.

If the Dashboard is already running, double-clicking the app simply reopens [http://localhost:8501](http://localhost:8501); it does not start a duplicate service or another listener. Closing the browser does not necessarily stop Streamlit. Double-click the app again to reopen the Dashboard.

If `.venv` is missing, choose **Run Setup** in the native dialog. This opens the existing `setup_amazon_review_analyzer.command`; the app does not install dependencies itself. If Ollama or `qwen3:8b` is unavailable, choose **Open Limited Mode** to view existing content without inference.

To stop a Dashboard started by the native app, double-click `launcher/Stop Amazon Review Analyzer.command`. It validates the saved project PID before stopping only that Streamlit process; it does not stop Ollama or any other Python process.

### Compatibility `.command` launcher

The existing `Amazon Review Analyzer.command` remains available for a Terminal-visible launch and shares the project launcher logic. It is useful when you explicitly want to keep the launching Terminal open and use `Ctrl+C` to stop that instance.

If `.venv` or a critical dependency is missing, the launcher directs you to `setup_amazon_review_analyzer.command` instead of showing a Python traceback.

### Start from Terminal

```bash
cd /path/to/amazon_review_analyzer
source .venv/bin/activate
streamlit run app.py
```

The browser normally opens [http://localhost:8501](http://localhost:8501). Press `Ctrl+C` in the launching Terminal to stop that Dashboard process. This does not delete data or stop Ollama.

## First Run

The first successful browser launch shows a single onboarding screen. Choose **English** or **简体中文**, review the local environment state, then enter Demo Dataset, Saved Analyses, or an available input route. Onboarding completion is saved only after you continue; creating or installing the `.app` does not pre-complete it. Later launches go directly to the Dashboard, and Quick Start remains available in the documentation.

## Limited Mode

The application still opens in **Limited Mode** when Ollama is missing, stopped, or does not contain `qwen3:8b`. You can still use:

- Demo Dataset;
- Saved Analyses;
- Competitor Comparison of saved results;
- existing reports and documentation.

Paste, Upload, and Amazon workflows that require new model inference are safely blocked. Existing results remain safe. After restoring Ollama, refresh System Status to continue or resume.

The Dashboard displays **Local AI unavailable — viewing features remain available.** This state is expected and is not an application failure.

## Amazon Quick Scan (optional)

Install the browser runtime only if you plan to use Amazon Quick Scan:

```bash
source .venv/bin/activate
playwright install chromium
```

Quick Scan reads only reviews available to an ordinary visible browser session. It does not bypass CAPTCHA, login, or other access controls. Use **Paste Reviews** or **Upload Dataset** when access is limited.

## Troubleshooting

### Ollama is not installed

Install it from the official Ollama website. Setup never installs system software silently. The demo and existing results remain available.

### Ollama is installed but not running

Open the Ollama application, wait for its local service to start, then refresh **System Status** or rerun Health Check.

### `qwen3:8b` is missing

```bash
ollama pull qwen3:8b
```

Refresh System Status after the download. Do not remove other local models.

### The Python environment or dependencies are missing

Double-click `setup_amazon_review_analyzer.command` in the project root again. Setup reuses a valid `.venv`; it does not delete user files or overwrite `.env`.

### Port 8501 is already in use

Open [http://localhost:8501](http://localhost:8501) first and check whether this project is already running. The desktop launcher reuses this project. If another application owns the port, close that application normally; the launcher does not kill unknown processes.

### The native app does not open

Confirm that the source project still contains `main.py`, `app.py`, `environment_health.py`, and `.venv`. If the project moved, run the project-local app or install it again so the private project-location preference can be refreshed. The launcher does not scan the whole Mac. For an unsigned development build, use Finder **Right-click → Open** once; never bypass Gatekeeper from Terminal.

### Analysis was interrupted

Production runs save checkpoints continuously. Restart the application and choose Resume. Do not edit checkpoint files manually.

### Resume a previous analysis

After restoring Ollama, choose **Resume** for the same current dataset. Existing `success` and `skipped` reviews are not analyzed again. **Retry Failed Reviews** processes only `failed` rows.

## Data and privacy

Analysis runs through local Ollama by default. Reviews, checkpoints, Saved Analyses, and reports stay in the local project workspace. Starting the Dashboard, browsing the demo, and running Health Check do not automatically contact Amazon or call OpenAI.

No OpenAI dependency or API key is required for the default configuration. Amazon Quick Scan is the only input route that contacts Amazon. Its availability depends on the current website, network, and browser session; Amazon may require login or human verification, limit pages, or return only some reviews. The app never bypasses CAPTCHA, login, anti-bot controls, or session restrictions. Use **Paste Reviews** or **Upload Dataset** when Quick Scan is unavailable.

Production analysis remains intentionally serial at **1 worker**. Runtime depends on Mac hardware, review length, and model cold start.

