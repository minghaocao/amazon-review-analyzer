# Step 16.5 — Native Mac App Experience

Status: `Native Mac App Experience: PASS` after the Step 16.5 acceptance checks complete.

Application version: **Amazon Review Analyzer v1.0.0**

Step 16.5 is an installation and launch-experience improvement. It is not a new product release, does not change the application version, and does not replace or modify the Step 15 release ZIP.

## Recommended launch

On macOS, double-click:

```text
dist/Amazon Review Analyzer.app
```

The launch path is deliberately small:

```text
Amazon Review Analyzer.app
→ shared launcher
→ existing environment health checks
→ local Streamlit Dashboard
→ default browser
```

The wrapper does not contain a Python runtime, `.venv`, Ollama, or `qwen3:8b`. Those components remain in their normal local locations. The app starts Streamlit in the background, waits for it to become ready, and opens `http://localhost:8501`.

If the same project is already running, a second launch reopens the existing Dashboard and does not create another listener. Closing the browser does not necessarily stop Streamlit; double-click the app again to reopen the page.

## First Run

The existing single-screen onboarding remains authoritative. Building or installing the native wrapper does not create or pre-complete first-run state. On the first real browser launch, the user can choose English or 简体中文, inspect local readiness, and enter Demo Dataset, Saved Analyses, or an available input workflow. Later launches do not force onboarding again.

## Limited Mode

Ollama or `qwen3:8b` is not required to open the Dashboard. When local AI is unavailable, the native dialog offers **Open Limited Mode / 打开有限模式**. Demo Dataset, Saved Analyses, existing competitor comparisons, existing reports, documentation, preview, and non-inference validation remain available. New Paste, Upload, or Amazon inference is disabled until local AI is ready.

The wrapper never installs Ollama, downloads a model, calls OpenAI, contacts Amazon, or runs review inference during launch.

## Setup and recovery

If `.venv` is missing, the native dialog offers **Run Setup / 运行设置** and opens the existing `setup_amazon_review_analyzer.command`. The wrapper does not duplicate dependency installation.

If port 8501 belongs to another application, the launcher does not kill or replace that process. It reports the conflict and stops safely.

To stop a Dashboard started by the native app, double-click:

```text
launcher/Stop Amazon Review Analyzer.command
```

The stop helper validates the saved PID, project working directory, Streamlit command, project signature, and health endpoint before signaling only that process. It never uses `killall`, never stops Ollama, and never stops an unrelated Python process.

## Privacy and security

- Runtime metadata is project-local and excludes reviews, Evidence, cookies, credentials, API keys, and environment dumps.
- Launcher logs contain only allowlisted operational fields and do not expose the user's absolute project path.
- Browser URLs are fixed localhost URLs with a validated numeric port.
- Commands use argument lists and quoted paths; no review or user input is interpolated into a shell command.
- The launcher does not request Accessibility, Screen Recording, or Full Disk Access.
- The launcher never removes quarantine attributes, disables Gatekeeper, changes SIP, or weakens macOS security settings.

## Gatekeeper and distribution status

The Step 16.5 `.app` is an **unsigned, not-notarized development wrapper**. It must not be described as Apple-signed or notarized. On first launch, macOS may require Finder **Right-click → Open** followed by confirmation. This is a known development-distribution limitation, not a reason to bypass Gatekeeper.

## Compatibility launcher

The existing `Amazon Review Analyzer.command` remains available as a Terminal-visible compatibility path and shares the project launcher logic. Manual `streamlit run app.py` also remains supported for development.

## Scope confirmation

Step 16.5 does not modify model prompts, taxonomy, review analysis semantics, evaluator metrics, aggregation, keyword analysis, Amazon parsing or scraping, frozen evaluation artifacts, Demo data, Saved Analyses, comparison outputs, report outputs, performance assets, or the Step 15 release package.

Final release packaging remains a separate future maintenance-release decision. The application version remains **1.0.0**.

## Acceptance record

Validated locally on 2026-10-22:

- Native wrapper: built successfully at `dist/Amazon Review Analyzer.app`.
- Bundle identifier: `com.caominghao.amazonreviewanalyzer`.
- Distribution status: unsigned and not notarized.
- Environment: Python 3.13.14, Streamlit 1.64.0, Ollama 0.40.6, and local `qwen3:8b` present.
- Real launch: Streamlit became ready on port 8501 and the browser opened the Dashboard without a Terminal-based primary experience.
- Duplicate launch: the existing Dashboard was reused; exactly one 8501 listener remained.
- Safe stop and relaunch: passed using verified project runtime metadata.
- Dashboard smoke: Demo, all five analysis tabs, Saved Analyses, existing report entry, Competitor Comparison, Paste, Upload, and Amazon Quick Scan entry points rendered without running inference.
- Localization smoke: English → 中文 → English passed while Demo data remained selected and unchanged.
- Limited Mode and First Run: passed with isolated offline UI profiles and temporary first-run state.
- Automated tests: 1178 / 1178 passed.
- Release health: 8 / 8 checks passed.
- Protected assets: 195 files checked; 0 unexpected SHA-256 mismatches.
- Step 15 ZIP: byte-identical, SHA-256 `8755f81b66fe677bd2babece50b7c69edcb70accf58d4b70256e616653d00d4e`.
- New LLM calls: 0.
- Amazon, OpenAI, and external-download access: none.
