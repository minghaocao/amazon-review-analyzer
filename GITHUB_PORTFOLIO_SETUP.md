# GitHub Portfolio Setup

Use this checklist after the Step 18.1 documentation changes have been reviewed and committed. These are manual GitHub presentation tasks; none of them are required to run the application.

## 1. Pin the repository

1. Open your GitHub profile.
2. In **Pinned**, choose **Customize your pins**.
3. Select `amazon-review-analyzer`.
4. Place it near the front of the pinned list.

## 2. Set the repository description

Recommended About description:

> Local AI-powered Amazon review intelligence with aspect sentiment analysis, evidence validation, competitor comparison, bilingual dashboards and business reports.

Keep the description factual and do not describe the project as affiliated with Amazon.

## 3. Verify Topics

Use only topics that describe implemented capabilities. Recommended set:

- `python`
- `streamlit`
- `ollama`
- `qwen`
- `llm`
- `nlp`
- `sentiment-analysis`
- `aspect-based-sentiment-analysis`
- `ecommerce`
- `amazon-reviews`
- `customer-insights`
- `business-intelligence`
- `data-analysis`

GitHub path: repository page → **About** → gear icon → **Topics**.

## 4. Upload the Social Preview

The repository-owned image is:

`assets/github-social-preview.png`

To upload it:

1. Open the repository on GitHub.
2. Go to **Settings** → **General**.
3. Find **Social preview**.
4. Choose **Edit** / **Upload an image**.
5. Upload `assets/github-social-preview.png`.
6. Save, then verify the thumbnail is readable.

The image is 1280 × 640, uses an original project design, and contains no Amazon logo or affiliation claim.

## 5. Keep v1.0.1 as Latest

Open **Releases** and confirm **v1.0.1** remains marked **Latest**. Step 18.1 does not create a tag or release.

## 6. Add the project to portfolio materials

- Add the public repository URL to the Projects section of your resume or portfolio.
- Optionally add it to LinkedIn under **Projects**.
- Link to the README for the product overview.
- Link to `docs/PORTFOLIO_CASE_STUDY.md` when a longer engineering narrative is useful.
- Describe the evaluation precisely as: “85.66% Aspect + Sentiment F1 on a frozen 150-case Blind V2 evaluation set.”

## 7. Final public-page review

- Confirm the principal README screenshot renders.
- Open every Documentation link.
- Confirm the v1.0.1 release link works.
- Confirm the MIT license is detected.
- Confirm no local path, token, `.env`, cookie, browser profile, or user dataset is visible.
- Confirm the Amazon independence disclaimer is visible.
- Review the repository page while signed out or in a private browser window.

## Suggested interview one-liner

> I built a local-first review-intelligence product that turns unstructured customer feedback into aspect-level sentiment, exact-source evidence, deterministic business insights, competitor comparisons, and bilingual reports.
