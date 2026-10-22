# Amazon Review Analyzer v1.0.0 — 中文快速开始

Amazon Review Analyzer 在 macOS 本机把商品评论整理为可追溯的 Aspect、Sentiment、Evidence、关键词和商业洞察。

> 最快体验方式：直接打开应用并使用 **Demo Dataset**。浏览 Demo 不需要 Ollama、模型或 Amazon 网络访问。

本快速开始配合应用源码项目使用。请在包含 `requirements.txt`、`.env.example` 和 `app.py` 的源码项目根目录执行安装与启动命令；Portfolio ZIP 本身不是独立源码发行包。

## 1. 环境要求

- macOS
- Python 3.11 或更高版本（v1.0 已在 Python 3.13.14 环境完成验收）
- Ollama 和模型 `qwen3:8b`（只有分析新评论时需要，需单独安装）
- Amazon Quick Scan 另需 Playwright Chromium

Release 包不包含 `.venv` 或模型文件。

## 2. 首次设置

先从 [Ollama 官方网站](https://ollama.com/download/mac) 安装并启动 Ollama，然后在 Terminal 中进入项目根目录：

```bash
cd /path/to/amazon_review_analyzer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
test -f .env || cp .env.example .env
ollama pull qwen3:8b
python main.py --check-ollama
```

如果已经存在 `.env`，不要覆盖它。默认的本地 `Ollama + qwen3:8b` 不需要 OpenAI API key。

只有准备使用 **Amazon Quick Scan** 时，才需要额外安装浏览器运行环境：

```bash
source .venv/bin/activate
playwright install chromium
```

## 3. 启动应用

### 方法 1：桌面启动器（推荐日常使用）

完成首次设置后，双击桌面的：

```text
Amazon Review Analyzer.command
```

启动器会进入已配置的项目目录、激活 `.venv`、检查 Ollama 和模型、避免重复启动 Streamlit，并打开 Dashboard。Ollama 未运行时，它会有限次尝试打开 Ollama 应用。

即使 Ollama 或模型不可用，Demo Dashboard 仍可正常打开；此时只是不能分析新的评论。按 Terminal 中的 `Ctrl+C` 可停止本次 Dashboard，不会删除数据或停止 Ollama。

> 项目移动后需同步修改启动器中的 `PROJECT_DIR`。macOS 首次拦截时，可在 Finder 中右键文件并选择 **打开**。

### 方法 2：Terminal 手动启动

以后每次手动启动只需：

```bash
cd /path/to/amazon_review_analyzer
source .venv/bin/activate
streamlit run app.py
```

浏览器通常会自动打开 [http://localhost:8501](http://localhost:8501)。应用默认显示 Demo Dataset，不会因为启动而自动调用模型或访问 Amazon。

## 4. 选择输入方式

在侧边栏的 **Dataset Mode** 中选择数据来源：

### Option A — Demo Dataset

最快、最稳定的体验方式。冻结 Demo 含 15 条评论、14 条成功分析，Coverage 为 93.3%，可离线浏览全部五个 Dashboard 页面。

### Option B — Paste Reviews

进入 **Analyze New Dataset → Paste Reviews**，每个非空行粘贴一条评论；预览确认后再开始本地 AI 分析。

### Option C — Upload Dataset

进入 **Analyze New Dataset → Upload Dataset**，上传不超过 10 MB 的 CSV。界面会让你映射字段：

- 评论正文 `content`：必需
- `review_id`、`rating`、`title`、`date`、`verified_purchase`：可选或推荐

缺少 Review ID 时会生成稳定 ID。评分应为 1–5；没有评分时，评分相关视图会隐藏。

### Option D — Amazon Quick Scan

进入 **Analyze New Dataset → Amazon Quick Scan**，输入受支持 Amazon marketplace 的商品 URL：

1. 选择 **Open Amazon Product**，打开普通可见浏览器。
2. 如 Amazon 要求登录或人工验证，请由你本人在浏览器中完成。
3. 选择 **Scan Available Reviews**，先检查可访问评论的预览；此时不调用模型。
4. 确认后选择 **Analyze Available Reviews**，再运行本地 AI 分析。

Quick Scan 只能读取当前普通浏览器会话可访问的评论，不能保证取得全部评论，也不会绕过 CAPTCHA、登录或其他访问控制。受限时请改用 **Paste Reviews** 或 **Upload Dataset**。

## 5. 分析完成后

Dashboard 包含 **Overview、Product Insights、Customer Language、Aspect Deep Dive、Review Explorer**。完整结果可保存到本机 **Analysis Library**，之后可重新打开、生成独立 HTML 报告，或选择两款商品进行 Competitor Comparison。

## 6. 常见问题

- **提示找不到虚拟环境**：在项目根目录重新执行 `python3 -m venv .venv` 和依赖安装步骤；启动器不会自动创建环境。
- **提示本地 AI 不可用**：启动 Ollama，运行 `ollama list` 确认存在 `qwen3:8b`，再运行 `python main.py --check-ollama`。
- **8501 端口已占用**：打开 `http://localhost:8501`，确认是否已有本项目实例；桌面启动器会复用它。
- **Amazon 浏览器无法启动**：激活 `.venv` 后运行 `playwright install chromium`；仍不可用时使用 CSV 或粘贴输入。
- **Amazon 要求验证或访问受限**：由用户本人完成合法验证；如仍受限，使用 Upload / Paste。不要尝试绕过网站访问控制。

默认分析在本机 Ollama 中完成，数据和报告保存在本机项目目录。只有 Amazon Quick Scan 会主动访问 Amazon。
