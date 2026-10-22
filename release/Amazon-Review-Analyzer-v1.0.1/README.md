# Amazon Review Analyzer v1.0.1

**Status: Stable Local Release**  
**Release type: Maintenance / UX / Distribution Release**

Amazon Review Analyzer 是一款在 macOS 本机运行的 Amazon 商品评论分析工具。它把评论整理为可追溯的 Aspect、Sentiment、Evidence、关键词和商业洞察，并可保存分析、比较两款商品，以及导出独立的中英文 HTML 报告。v1.0.1 将已验收的 macOS `.app` 启动器、首次运行引导、Environment Health、Limited Mode、Diagnostics、安装助手与安全停止助手正式纳入稳定本地发布；分析模型和业务语义没有改变。

The macOS `.app` is a lightweight launcher and **not a fully self-contained application**. It depends on the configured local project, Python environment, Ollama, and `qwen3:8b` for new AI analysis.

默认分析引擎是本地 **Ollama + `qwen3:8b`**。内置 Demo Dataset 不需要 Ollama 或网络即可浏览；只有用户主动分析新评论时才会调用本地模型。

> English and Simplified Chinese are available in the application. Review text, keywords, and Evidence always remain in their original language.

## What it does

The application turns imported product reviews into traceable Aspect, Sentiment, and Evidence records, then aggregates those records into customer concerns, business insights, customer language, saved comparisons, and standalone reports.

## Architecture

程序接收 Demo、CSV、粘贴文本或当前 Amazon 浏览器会话可访问的评论，完成以下流程：

```text
Amazon Quick Scan / Paste / Upload
→ Validation
→ Cleaner
→ qwen3:8b via Ollama
→ Aspect + Sentiment + Evidence
→ Aggregator
→ Keyword Analyzer
→ Business Intelligence
→ Dashboard
→ Saved Analysis
→ Competitor Comparison
→ Reports
```

每个分析结果都会保留原评论，并把产品属性与针对该属性的情绪分开。例如，一条评论可以同时包含 Picture Quality 的正面评价和 Remote Control 的负面评价。Evidence 必须是原评论中的连续原文片段，不能由模型改写。

## Features

- **四种输入方式**：Demo Dataset、Amazon Quick Scan、Upload Dataset、Paste Reviews。
- **本地 AI 分析**：使用 `qwen3:8b` 提取 Summary、Aspect、Sentiment 和原文 Evidence。
- **15 个 canonical Aspects**：Picture Quality、Sound Quality、Price / Value、Remote Control、System Performance、Ease of Use、Design、Build Quality、Reliability、Connectivity、Setup / Installation、Customer Service、Delivery、Packaging 和 Other。
- **聚合与客户语言**：Selling Points、Pain Points、Opportunity Ranking、Priority Matrix、Rating Insights、Hidden Complaints、关键词和 2-gram / 3-gram 短语。
- **稳定批处理**：正式模式固定为 1 worker，提供进度、elapsed time、动态 ETA、checkpoint、Resume、Retry Failed Reviews，以及仅针对瞬时 provider 连接错误的有限重试。
- **五个 Dashboard 页面**：Overview、Product Insights、Customer Language、Aspect Deep Dive、Review Explorer。
- **中英文界面**：English / 简体中文可随时切换，不重新分析数据，不翻译客户原文。
- **Saved Analyses**：带 SHA-256 完整性检查的本地快照库，重启后仍然保留。
- **两产品对比**：Product A vs Product B、固定 15 个 Aspect、Competitive Advantages、Competitive Weaknesses、White-Space Opportunities、Rating、Customer Language 和 Evidence 对比。
- **商业报告**：单产品及竞品对比报告均支持 English / 中文，导出为完全独立的 HTML，可在浏览器中 Print / Save as PDF。

## Requirements

v1.0.1 的正式桌面验收环境是 macOS。建议准备：

- macOS
- Python 3.11 或更高版本
- [Ollama](https://ollama.com/download/mac)
- 本地模型 `qwen3:8b`
- 约 10 GB 或以上可用磁盘空间，用于 Python 环境、Playwright Chromium 和本地模型
- 适合运行 8B 本地模型的内存；当前验收机器为 16 GiB Apple Silicon Mac

Amazon Quick Scan 额外需要 Playwright Chromium。没有 Chromium、Ollama 或网络时，Demo Dashboard、Saved Analyses 和已生成报告仍然可以使用。

## Quick Start

### First-Time Setup

首次使用时，在 Finder 中打开项目目录并双击：

```text
setup_amazon_review_analyzer.command
```

Setup 会检查 Python 3.11+、创建缺失的 `.venv`、从现有 `requirements.txt` 安装依赖，并检查 Ollama 与 `qwen3:8b`。它不会安装 Ollama、自动下载大型模型、访问 Amazon、运行评论推理、删除现有 `.venv` 或覆盖现有 `.env`。

完成后可检查结构化环境状态：

```bash
source .venv/bin/activate
python main.py --health-check
```

退出码为 `0` 表示 Ready，`1` 表示 Action Required / Limited，`2` 表示 Unavailable（核心环境无法安全运行）。需要 machine-readable 输出时使用：

```bash
python main.py --health-check-json
```

Health Check 只进行本地文件、依赖、进程、端口和 Ollama metadata 检查；不会执行模型推理或访问 Amazon。

如果 Ollama 或模型尚未准备好，Dashboard 仍会以 **Limited Mode** 打开。Demo Dataset、Saved Analyses、已有 comparison、reports 和文档仍可使用；需要新模型推理的分析入口会被安全阻止。

然后双击 `dist/Amazon Review Analyzer.app`，并从 Demo Dataset 开始体验。更完整的首次安装说明见：

- [中文快速开始](QUICK_START_CN.md)
- [English Quick Start](QUICK_START_EN.md)

如果要使用 Amazon Quick Scan，再明确安装普通 Chromium runtime：

```bash
source .venv/bin/activate
playwright install chromium
```

手动启动方式：

```bash
source .venv/bin/activate
streamlit run app.py
```

Streamlit 通常会自动打开 [http://localhost:8501](http://localhost:8501)。应用默认进入 Demo Dataset，不会因为启动而自动调用 Ollama 或 Amazon。

运行离线自动测试：

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
```

自动测试使用 mock 和本地 fixture，不应访问真实 Ollama、OpenAI 或 Amazon。冻结的 Blind / Holdout 历史评估不属于日常测试，不要重新运行。

## One-click launcher

### Recommended Launch on macOS

推荐双击项目中的：

```text
dist/Amazon Review Analyzer.app
```

这是一个轻量 macOS launcher wrapper，而不是自包含二进制。它不会打包 Python、`.venv`、Ollama 或 `qwen3:8b`；启动链路为：

```text
Amazon Review Analyzer.app → shared launcher → health check → local Streamlit → browser
```

正常启动不会把 Terminal 作为主界面。App 在后台启动本地 Dashboard，等待 `http://localhost:8501` 就绪后只打开一次默认浏览器。如果本项目已经运行，再次双击只会重新打开现有页面，不会创建第二个 Streamlit listener。关闭浏览器不一定会停止本地服务；再次双击 App 可重新打开 Dashboard。

首次真正进入 Dashboard 时会显示 onboarding，可选择 English / 简体中文并进入 Demo、Saved Analyses 或可用输入入口。创建 `.app` 不会预写 first-run 完成状态。

若 `.venv` 缺失，原生提示会提供 **Run Setup / 运行设置**，并调用已有 `setup_amazon_review_analyzer.command`。若 Ollama 或 `qwen3:8b` 不可用，可选择 **Open Limited Mode / 打开有限模式**；Demo、Saved Analyses、已有 comparison 与 reports 仍可查看，新推理入口会被安全禁用。

要安全停止由 `.app` 启动的 Dashboard，可双击：

```text
launcher/Stop Amazon Review Analyzer.command
```

停止器只在 PID、项目目录、Streamlit command 与 health endpoint 均验证通过后停止本项目进程；不会使用 `killall`，不会停止 Ollama 或其他 Python 程序。

当前 v1.0.1 本地发布版 `.app` 未签名、未 notarize。首次被 Gatekeeper 拦截时，可在 Finder 中右键 App 并选择 **Open / 打开**，再确认系统提示。不要执行 quarantine 绕过命令，不要关闭 Gatekeeper、SIP 或其他 macOS 安全机制。

## Compatibility `.command` launcher

macOS 桌面启动器位于：

```text
~/Desktop/Amazon Review Analyzer.command
```

双击后，它会：

1. 进入正式项目目录并激活 `.venv`。
2. 执行轻量环境检查，不进行模型推理。
3. 检查 Ollama 命令、服务和 `qwen3:8b`。
4. Ollama 已安装但服务未运行时，有限次尝试打开 Ollama 应用。
5. 检查本项目 Streamlit 是否已经运行；若已运行，只打开现有页面，不创建第二个实例。
6. 在 `http://localhost:8501` 启动 Dashboard 并打开浏览器。

即使 Ollama 没有安装、没有运行或模型不存在，启动器也会继续打开 Demo Dashboard，并提示新 AI 分析暂不可用。Terminal 中按 `Ctrl+C` 只停止本次启动的 Streamlit，不删除任何数据，也不停止 Ollama。

如果 `.venv` 或关键依赖缺失，启动器会清楚提示先运行项目根目录的 `setup_amazon_review_analyzer.command`，不会把 Python traceback 显示给普通用户。

`.app` 使用有界 project discovery：优先检查 bundle 相对位置和私有本地偏好，再检查当前用户的标准开发目录，不会扫描整台电脑。`.command` 继续作为 Terminal-visible 的兼容入口。首次被 macOS 安全机制拦截时，可在 Finder 中右键启动器并选择 **Open / 打开**；不要关闭系统安全检查。

## English / 中文

侧边栏顶部的 **Language / 语言** 可以在 English 和简体中文之间切换。

- 默认语言是 English。
- 切换语言不会调用 Ollama，不会重新聚合，也不会丢失当前 Dataset Mode 或已有分析。
- UI 文案和确定性商业总结可本地化。
- Aspect 名称、sentiment、CSV 列名和 JSON key 继续使用英文 canonical schema，以保持兼容性。
- Original Review、Evidence、关键词和客户短语始终保留原始语言。
- 报告语言可独立于 Dashboard 语言选择。

## Input Methods

### Demo Dataset

应用启动后的默认数据。冻结的 Demo 包含 15 条评论，其中 14 条成功分析，Coverage 为 93.3%。它用于无网络、无 Ollama 时稳定展示全部 Dashboard 页面。

### Paste Reviews

在 **Analyze New Dataset → Paste Reviews** 中每行粘贴一条评论。程序会生成稳定的 Review ID，先预览、验证和清洗，再由用户明确开始分析。

### Upload Dataset

在 **Analyze New Dataset → Upload Dataset** 中上传最大 10 MB 的 CSV。界面允许把评论正文、ID、评分、标题、日期和 verified purchase 映射到标准字段；缺少 ID 时会生成稳定 ID。评论正文是必需数据，缺少可选列不会导致整个文件失败。

标准 CSV 字段为：

| Field | Requirement | Notes |
| --- | --- | --- |
| `content` | Required | 原始评论正文；不能为空 |
| `review_id` | Recommended | 缺失时可由交互式导入生成 |
| `rating` | Recommended | 数值应在 1–5；无评分时相关 Dashboard 区域隐藏 |
| `title` | Optional | 空标题允许 |
| `date` | Optional | 保留原始日期值 |
| `verified_purchase` | Optional | 仅作为 metadata，不代表可信度分数 |

### Amazon Quick Scan

在 **Analyze New Dataset → Amazon Quick Scan** 中输入受支持 marketplace 的商品 URL，然后：

1. **Open Amazon Product** 打开普通可见 Chromium。
2. 如 Amazon 要求登录或人工验证，由用户本人在浏览器中完成。
3. **Scan Available Reviews** 读取当前浏览器 session 可以合法访问到的评论并显示预览；此时不调用模型。
4. 确认后点击 **Analyze Available Reviews**，才保存原始评论并运行分析。

支持 `amazon.com`、`amazon.co.uk`、`amazon.de`、`amazon.fr`、`amazon.it` 和 `amazon.es`，以及常见 `/dp/ASIN`、`/gp/product/ASIN`、`/gp/aw/d/ASIN`、`/product-reviews/ASIN` 和 `/portal/customer-reviews/ASIN` 路径。

## Amazon Quick Scan limitations

Amazon may require login or limit automated access. Quick Scan 只分析当前普通浏览器会话实际可访问的评论，不承诺稳定获取商品的全部评论，也不把 Amazon 页面显示的总评论数当作已导入数量。

- 不绕过 CAPTCHA、Robot Check、登录、Access Denied 或其他访问控制。
- 不使用 stealth、指纹伪装、验证码服务或账户自动登录。
- 可访问到 8 条时，可以分析这 8 条；Coverage 按成功分析数除以实际导入数计算。
- 登录受限、access limited、无评论或页面结构变化会分别显示友好状态，而不是 traceback。
- 受限时可以直接切换到 Paste Reviews 或 Upload Dataset。
- Amazon 网站结构和访问策略可能随时变化；稳定演示请使用 Demo Dataset。

Amazon Quick Scan analyzes reviews currently accessible from an Amazon product page. For larger or controlled datasets, use Upload Dataset or Paste Reviews.

## Analysis and runtime

### Aspect, sentiment, and Evidence

`analyze_review()` 返回 Summary 和一组 Aspect mentions。每个 mention 包含原始属性表达、canonical Aspect、`positive` / `negative` / `neutral` sentiment，以及经验证的 Evidence 原文片段。Python 后处理根据已验证的 Evidence 派生 positive / negative points，并保留同一评论中的原始多次 mention。

### Aggregator and Keyword Analyzer

Aggregator 按不同评论统计 PRODUCT、EXPERIENCE 和 OTHER，避免同一评论重复 mention 被错误当作多位顾客。Delivery、Packaging 和 Customer Service 单独属于 Shopping Experience，不会自动成为产品质量 Selling Point 或 Pain Point。

Keyword Analyzer 离线读取已完成分析，统计有商业含义的 unigram、bigram 和 trigram，并保留 Review 引用。它不会再次调用模型或推断新的 sentiment。

### Stable runtime

- 正式生产模式固定为 **Stable mode / 1 worker**。
- 每完成一条评论都会更新 production checkpoint。
- 进度显示当前 Review ID、成功/失败/跳过数及 elapsed time。
- 至少有足够的近期样本后才计算动态 ETA；数据不足时显示 **Estimating...**。
- Resume 不重新分析已有 `success` 或 `skipped` 行。
- Retry Failed Reviews 只重新处理 `failed` 行，成功后更新聚合、关键词和 Dashboard outputs。
- 仅明确的瞬时 provider / connection failure 会等待约 1 秒并最多自动重试一次；Evidence、schema、内容质量或用户输入错误不会自动重试。

Upload / Paste 使用 `outputs/session/current/`，Amazon 使用 `outputs/amazon/current/`，冻结 Demo 使用项目根 `outputs/`。三套数据不会互相覆盖。

## Dashboard

### Overview

显示 Total Reviews、Coverage、Executive Summary、Aspect Mention Sentiment、Selling Points、Pain Points 和 Customer Concerns。

### Product Insights

显示 Opportunity Ranking、Priority Matrix、Product vs Shopping Experience、Rating Insights、Hidden Complaints 和确定性 Business Recommendations。

### Customer Language

显示高频关键词、正负短语以及来自已验证 Evidence 的 Aspect phrases。

### Aspect Deep Dive

显示 Review Share、Mention Count、sentiment rates、Average Rating、Evidence Level、关键词及可追溯原评论。

### Review Explorer

支持 keyword、rating、status、Aspect、sentiment、verified purchase、Product Issues 和 Shopping Experience 筛选，并可返回每条评论的 Original Review、Summary、Aspect、Sentiment 和 Evidence。

## Saved Analyses

完整分析可通过 **Save Analysis / 保存分析结果** 保存到本机 Analysis Library。每个快照包含产品名称、稳定 Analysis ID、来源、样本数、Coverage、版本 metadata、七个 Dashboard 必需文件及关键文件 SHA-256。

- 快照位于 `outputs/library/<safe-slug>_<analysis-id>/`。
- Saved Analysis 是磁盘持久化数据，不只存在于浏览器 session；Streamlit 重启后仍可加载。
- 相同 analysis run 不会被静默重复保存。
- 加载、比较或生成报告前会验证 schema、candidate、taxonomy 和 hash。
- 文件损坏或版本不兼容时会安全停止，不会忽略完整性错误，也不会重新调用模型。
- Saved Analyses 只保存在这台电脑上，不自动上传到云端。

## Competitor Comparison

**Compare Products / 竞品对比** 从两项不同且兼容的 Saved Analyses 读取已有结果，不调用 Qwen，也不重跑 Aggregator 或 Keyword Analyzer。

- V1 只支持 Product A vs Product B。
- 使用固定的 15 个 canonical Aspects。
- 显示 Competitive Advantages、Competitive Weaknesses、White-Space Opportunities、Rating、Customer Language 和 Evidence Comparison。
- 所有方向结论都保留样本数量和 Evidence Level；样本小或不均衡时显示警告。
- 一侧没有 Aspect 数据时显示 **No evidence / 暂无证据**，不伪装成 `0% positive`。
- 不给出没有依据的 overall winner，也不声称统计显著性。
- 相同 A/B 的核心确定性业务结果可复现；生成 ID 和时间戳可以不同。

比较结果写入 `outputs/comparison/current/`，不会覆盖 Saved Analysis 或 Demo。

## Report Export

可以从完整 Dashboard、Saved Analyses 或已完成的 A/B comparison 生成：

- Single Product Report
- Competitor Comparison Report
- English HTML
- 简体中文 HTML

报告写入 `outputs/reports/<report-id>/report.html`，并附带 `report_metadata.json`。HTML：

- 内嵌全部 CSS，不依赖 Streamlit、localhost、CDN、外部字体、图片或 JavaScript。
- App 关闭后仍可直接双击阅读。
- 支持浏览器 **Print → Save as PDF**。
- 动态内容统一 HTML escape，产品名称或 Evidence 中的标签只按文本显示。
- 不包含项目绝对路径、`.venv`、临时目录或本机用户名。
- 保留原始 Customer Evidence 和客户短语，不自动翻译。
- Saved snapshot 及 comparison 来源会在导出前通过完整性检查。

生成报告只读取现有输出，不调用 Ollama、Amazon、Aggregator 或 Keyword Analyzer。

## Accuracy and evaluation context

不要把单一评估结果理解为所有商品、语言和评论风格下的通用准确率。

冻结的 Blind V2 评估结果：

| Metric | Frozen result |
| --- | ---: |
| Cases | 150 |
| Aspect + Sentiment F1 | 85.66% |
| Aspect-only F1 | 87.11% |
| Sentiment Accuracy | 98.33% |
| Evidence Validity | 100% |
| Evidence Match | 100% |

These metrics apply only to the frozen Blind V2 evaluation set and do not represent universal accuracy. 它们不是“该应用准确率为 85.66%”的营销声明，也不代表统计置信区间或任意真实 Amazon 数据集的预期表现。

Aspect + Sentiment F1 = 85.66% on the frozen 150-case Blind V2 evaluation. 该数字不应表述为通用准确率。

历史 Blind、Holdout、Final Blind 和 development evaluation assets 已冻结。日常启动、测试或发布验收只校验其文件完整性，不重新运行模型评估，也不根据这些结果继续调 Prompt、taxonomy 或 post-processing。

## Privacy

默认隐私边界：

- `qwen3:8b` 通过本机 Ollama 推理；默认不需要 OpenAI API key。
- 原始评论、checkpoint、Current Analysis、Saved Analyses、comparisons 和 reports 均存储在本机项目目录。
- Dashboard 浏览、语言切换、筛选、Saved Analysis 打开、comparison 和 report export 不触发隐藏模型调用。
- 运行事件日志只保存 Review ID、状态、耗时、retry 和 error type 等必要 metadata，不保存完整评论正文。
- Amazon Browser Import 会访问 Amazon 网站，这是唯一需要访问 Amazon 的输入模式；普通 Streamlit Dashboard 和 Demo 不会因为启动而访问 Amazon。
- 可选 OpenAI provider 仅在用户明确配置并选择后使用；不要把 API key 写入代码或提交 `.env`。
- Amazon 浏览器 session 的 cookies、密码和 token 不写入 Analysis Library 或报告。

## Known Limitations

- Amazon 可能要求登录、人工验证、challenge browser session、返回部分评论或限制可访问页面；Quick Scan 无法保证导入全部评论。系统不绕过 CAPTCHA、登录、Amazon anti-bot controls 或 session restrictions；受限时使用 Paste Reviews 或 Upload Dataset。
- `qwen3:8b` 是本地 8B 模型，长评论和大量评论在 16 GiB Mac 上可能需要较长时间。冻结的 20-review 性能基准中，Stable 1-worker 的中位延迟约为 12.14 秒/条；真实速度取决于硬件、评论长度和模型冷启动。
- 新 AI 分析需要本机 Ollama 与 `qwen3:8b`；正式稳定运行模式仍为 **1 worker**。默认不需要 OpenAI API。
- 当前 `.app` 是依赖现有项目目录与本地环境的 lightweight launcher，不内置 Python runtime、`.venv`、Ollama、`qwen3:8b` 或完整 backend；它不是完全自包含的二进制。
- `.app` 未为公开分发 code sign 或 notarize；在其他 Mac 上可能需要 Finder 中的 **Right Click → Open**。
- 小样本结论是不稳定的方向性信号。1 条或少量评论可以运行，但正式排行榜可能显示 **Insufficient Evidence**。
- Selling、Pain、Opportunity 和 competitor comparison 是透明的业务启发式，不是统计显著性声明。
- 竞品结论取决于 A/B 数据量、时间范围、来源和样本质量是否可比；不均衡样本会降低可解释性。
- Evidence、原评论和 Customer Language 保留原始语言；中文 UI 不会自动翻译英文客户原话。
- Aspect taxonomy 固定为 15 个 canonical categories；细分类目特征可能落入 Other。
- v1.0 competitor comparison 仅支持两个已保存商品，不支持多产品比较、云部署、用户账户或共享数据库。
- HTML 报告可以通过浏览器另存 PDF，但项目本身不生成原生 PDF 文件。

## Troubleshooting

### `Python virtual environment was not found`

双击项目根目录的 `setup_amazon_review_analyzer.command`。Setup 会创建缺失的 `.venv` 并从 `requirements.txt` 安装依赖；它不会删除已有环境、覆盖 `.env` 或下载大型模型。

### Some required Python packages are missing

重新运行 `setup_amazon_review_analyzer.command`。若安装仍失败，请检查网络与 Python 3.11+，然后运行：

```bash
source .venv/bin/activate
python main.py --health-check
```

### Ollama not installed

从 [Ollama 官方网站](https://ollama.com/download/mac) 安装。Setup 不会静默安装系统软件。Demo、Saved Analyses、已有 comparison 和 reports 仍可使用。

### Ollama not running

打开 Ollama 应用，等待本地服务启动，然后刷新 Dashboard 的 **System Status** 或重新运行 Health Check。桌面启动器仅按现有机制进行有限次启动尝试。

### `qwen3:8b` missing

确认有足够本地空间后明确运行：

```bash
ollama pull qwen3:8b
```

下载完成后刷新 System Status。Setup 不会自动下载模型。

### Port 8501 is already in use

先打开 [http://localhost:8501](http://localhost:8501) 检查是否已经是本项目。桌面启动器会复用本项目已有实例；如果端口属于其他应用，请先正常关闭该应用，不要强杀 Ollama。

### Amazon 浏览器无法启动

```bash
source .venv/bin/activate
playwright install chromium
```

仍失败时使用 Upload Dataset 或 Paste Reviews。不要关闭 TLS verification，也不要尝试绕过 Amazon 验证。

### Amazon 要求登录、验证或限制访问

在程序打开的普通可见浏览器中由用户本人完成要求的步骤，然后选择 **Continue After Verification**。若仍受限，切换到 Paste / Upload。这不代表评论分析模块失败。

### CSV 无法导入

确认文件不超过 10 MB，可以被 pandas 解码，并包含可映射的 Review Content。评分、标题、日期和 verified purchase 是可选字段；无有效正文时不会调用 Qwen。

### 分析速度较慢

保持 Ollama 运行以减少冷启动，避免在同一 16 GiB Mac 上并行运行多个 `qwen3:8b` 请求。正式模式有意保持 1 worker，以换取稳定性。大数据集会显示进度并可从 checkpoint Resume。

### 分析被中断 / Resume previous analysis

生产分析会持续保存 checkpoint。恢复 Ollama 并重新打开同一 Current Analysis 后选择 **Resume**；已有 `success` 与 `skipped` 评论不会重复分析。**Retry Failed Reviews** 只处理 `failed` 行。不要手工修改 checkpoint metadata。

### Resume 被拒绝

Checkpoint 与输入签名、模型和 analysis candidate 绑定。评论增删、换序、正文改变或版本不兼容时，旧 checkpoint 不可安全混用；请启动新分析，不要手工绕过检查。

### Saved Analysis 或报告完整性失败

快照文件被修改、缺失或 hash 不一致时，加载、comparison 和 report export 会被阻止。保留原快照并从可信的完整分析重新保存；不要手工改 manifest 来跳过校验。

### Dashboard 没有排行榜或关键词

这通常表示样本低于最小阈值，而不是程序错误。检查 Coverage、成功评论数和 empty-state 提示；原评论和 Review Explorer 仍可用。

## Project layout

核心模块职责：

| Module | Responsibility |
| --- | --- |
| `app.py` | Streamlit UI、输入模式、Dashboard 和用户操作编排 |
| `environment_health.py` / `first_run.py` | 确定性环境检查、Limited Mode gating、first-run 本地状态和安全 diagnostics |
| `input_loader.py` / `cleaner.py` | CSV 验证、原文保留和基础清洗 |
| `llm_client.py` / `review_analyzer.py` | provider 封装和单评论结构化分析 |
| `batch_analyzer.py` / `analysis_runtime.py` | 稳定批处理、checkpoint、Resume、retry 和进度 |
| `aggregator.py` / `keyword_analyzer.py` | 商品级统计和离线客户语言分析 |
| `business_intelligence.py` | 确定性商业指标和样本提示 |
| `browser_amazon_import.py` / `amazon_url.py` | 合规的普通浏览器导入和 URL normalization |
| `analysis_library.py` | Saved Analysis、manifest 和 snapshot integrity |
| `competitor_comparison.py` / `comparison_ui.py` | 确定性双产品对比 |
| `report_generator.py` | 独立、安全的中英文 HTML 报告 |
| `config.py` | 集中的版本、路径和运行配置 |

主要本地数据隔离：

```text
outputs/                       # frozen Demo outputs
outputs/session/current/       # current Upload / Paste analysis
outputs/amazon/current/        # current Amazon Quick Scan analysis
outputs/library/               # persistent Saved Analyses
outputs/comparison/current/    # current deterministic A/B comparison
outputs/reports/               # standalone exported reports
```

## Release verification

发布或面试前建议只执行离线检查和小规模 smoke test：

```bash
source .venv/bin/activate
python main.py --health-check
python main.py --check-ollama
python main.py --release-health-check
python -m unittest discover -s tests -v
streamlit run app.py
```

`--release-health-check` 只检查本地 release metadata、配置契约、翻译覆盖、冻结资产 SHA-256、报告模板、Saved Analysis 快照和桌面启动器；它不访问网络、不打开浏览器，也不调用模型。

然后人工确认：Demo 五个页面、English → 中文 → English、Paste / Upload 预览、Saved Analysis 持久化、两产品 comparison、English / 中文 standalone report 及 Chrome Print Preview。Amazon 验收应尊重实时网站访问状态，遇到登录或限制时确认 fallback 即可，不应尝试绕过。

## About

- Product: **Amazon Review Analyzer**
- Version: **1.0.1**
- Release status: **Stable Local Release**
- Release type: **Maintenance / UX / Distribution Release**
- Default AI: local Ollama
- Model: `qwen3:8b`
- Runtime: Stable mode, 1 worker
- Storage: local project workspace
- UI languages: English / 简体中文

Amazon Review Analyzer v1.0.1 focuses on local, traceable review intelligence. It does not claim exhaustive Amazon access, universal model accuracy, a fully standalone application, or statistical significance.
