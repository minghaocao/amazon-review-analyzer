# Amazon Review Analyzer v1.0.1 Release Notes

**Release type:** Maintenance / UX / Distribution Release  
**Status:** Stable Local Release

Amazon Review Analyzer v1.0.1 formally packages the already accepted local desktop experience from Steps 16 and 16.5. It improves setup, startup, recovery, diagnostics, and local distribution without changing review-analysis behavior or business semantics.

## Added

- macOS `.app` launcher
- First Run onboarding
- Environment Health Check
- Limited Mode
- Advanced Diagnostics
- Install-to-Applications helper
- Safe Stop helper
- bilingual onboarding and diagnostics
- safer environment visibility before running new analysis

## Improved

- launcher single-instance handling
- startup and recovery UX
- clear separation between viewing features and features that require local AI
- guidance for missing Python environment, dependencies, Ollama, model, or browser runtime
- local installation and launch instructions

## Unchanged

- Analysis model unchanged
- `qwen3:8b` unchanged
- Prompt unchanged
- taxonomy unchanged
- sentiment logic unchanged
- Evidence validation unchanged
- Aggregator unchanged
- Keyword Analyzer unchanged
- Amazon access strategy unchanged
- Business Intelligence semantics unchanged
- competitor comparison semantics unchanged
- report semantics unchanged
- frozen evaluation results unchanged

No OpenAI call, Amazon request, or new `qwen3:8b` inference was required to create this maintenance release.

## Packaging statement

The macOS `.app` is a lightweight launcher for an existing local project installation. It is not a fully self-contained application and does not bundle Python, `.venv`, Ollama, `qwen3:8b`, or the complete backend. The release is intended as a portfolio and local-distribution package.

The app is not code signed or notarized for public distribution. Another Mac may require Finder **Right Click → Open**. Never disable Gatekeeper, SIP, certificate verification, or other platform security controls.

## Amazon access

Amazon Quick Scan depends on the current website, network, and browser session. Amazon may require login or human verification, challenge a session, expose only part of the available reviews, or restrict review pages. The application does not bypass CAPTCHA, login, anti-bot controls, or session restrictions. Paste Reviews and Upload Dataset remain the supported fallbacks.

## Local AI

New AI analysis requires a running local Ollama service and `qwen3:8b`. Runtime depends on Mac hardware, review length, and model state. The stable production configuration remains serial at 1 worker. The default configuration does not require OpenAI.

## Evaluation context

The frozen Blind V2 set contains 150 cases. Its frozen results are Aspect + Sentiment F1 85.66%, Aspect-only F1 87.11%, Sentiment Accuracy 98.33%, Evidence Validity 100%, and Evidence Match 100%.

These metrics apply only to the frozen Blind V2 evaluation set and do not represent universal accuracy. No frozen evaluation was rerun for v1.0.1.
