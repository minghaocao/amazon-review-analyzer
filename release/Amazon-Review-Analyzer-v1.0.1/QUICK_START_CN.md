# Amazon Review Analyzer v1.0.1 — 中文快速开始

**状态：Stable Local Release（稳定本地发布）**

Amazon Review Analyzer 是一款在 macOS 本机运行的商品评论分析工具。它将评论整理为可追溯的 Aspect、Sentiment、Evidence、关键词和商业洞察。

> 最快体验方式：打开应用后选择 **Demo Dataset**。Demo 不需要 Ollama、模型或 Amazon 网络访问。

## 推荐本地安装流程

1. 从官方来源安装 Ollama。
2. 确认本机已有 `qwen3:8b`；如果缺失，再明确执行 `ollama pull qwen3:8b`。
3. 在 Finder 中打开源码项目目录，双击 `setup_amazon_review_analyzer.command`。Setup 会检查 Python、在缺失时创建 `.venv`、安装 `requirements.txt` 中的依赖，并检查本地 AI 环境。
4. 当最终状态为 **READY** 后，可直接打开 `dist/Amazon Review Analyzer.app`，或使用安装助手复制到 Applications。
5. App 会启动本地 Streamlit UI，并打开 [http://localhost:8501](http://localhost:8501)。
6. 先从 **Demo Dataset** 开始，再分析新评论。

当前开发版 `.app` 未签名、未 notarize。若 Gatekeeper 首次阻止运行，请在 Finder 中右键该 App 并选择 **打开**，再确认 macOS 提示。不要删除 quarantine 属性、关闭 Gatekeeper 或降低系统安全设置。Setup 的 `.command` 文件首次运行时也可能需要使用正常的“右键 → 打开”。

该 `.app` 是轻量启动器，不是完全自包含的二进制。它需要源码项目、可用的项目环境和本地依赖；不内置 Python、`.venv`、Ollama、`qwen3:8b` 或完整 backend。

Setup 不会：

- 静默安装 Ollama；
- 自动下载体积较大的 `qwen3:8b`；
- 访问 Amazon；
- 运行评论分析；
- 删除现有 `.venv`；
- 覆盖现有 `.env`。

## 环境要求

- macOS
- Python 3.11 或更高版本
- Ollama 与 `qwen3:8b`（仅分析新评论时需要）
- Amazon Quick Scan 另需 Playwright Chromium

如果 Ollama 尚未安装，请从 [Ollama 官方网站](https://ollama.com/download/mac) 安装。Setup 只检测系统软件，不会代替用户安装。

如果 Setup 提示模型缺失，请在 Terminal 中明确执行：

```bash
ollama pull qwen3:8b
```

模型需要数 GB 本地空间，Setup 不会自动开始下载。

## 检查环境状态

激活项目环境后运行：

```bash
source .venv/bin/activate
python main.py --health-check
```

需要给脚本读取的结构化结果时，可运行：

```bash
python main.py --health-check-json
```

退出码：

| Code | Meaning |
| ---: | --- |
| `0` | Ready，可以启动并分析新评论 |
| `1` | Action Required / Limited，Dashboard 可用，但部分能力需要处理 |
| `2` | Unavailable，Python 环境或核心运行目录无法安全使用 |

Health Check 只检查本地文件、包、进程、端口和 Ollama metadata；不会运行模型，也不会访问 Amazon。

## 推荐启动方式

### macOS 原生启动器（推荐）

双击：

```text
dist/Amazon Review Analyzer.app
```

该 App 是一个轻量 launcher wrapper：它定位项目、复用现有环境健康检查、在后台启动本地 Streamlit、等待服务就绪，然后打开默认浏览器。它不会打包 Python、`.venv`、Ollama 或 `qwen3:8b`。

如果 Dashboard 已经运行，再次双击只会重新打开 [http://localhost:8501](http://localhost:8501)，不会创建第二个 Streamlit 服务或 listener。关闭浏览器不一定会停止 Streamlit；需要重新打开时，再次双击 App 即可。

如果缺少 `.venv`，可在原生提示框中选择 **Run Setup / 运行设置**，调用现有 `setup_amazon_review_analyzer.command`；App 自身不会安装依赖。如果 Ollama 或 `qwen3:8b` 不可用，可选择 **Open Limited Mode / 打开有限模式** 浏览已有内容。

需要停止由原生 App 启动的 Dashboard 时，双击 `launcher/Stop Amazon Review Analyzer.command`。它会先验证本项目保存的 PID，再只停止该 Streamlit 进程；不会停止 Ollama，也不会停止其他 Python 进程。

### 兼容 `.command` 启动器

原有 `Amazon Review Analyzer.command` 继续可用，并复用项目 launcher 逻辑。需要看到启动 Terminal、并通过 `Ctrl+C` 停止本次实例时，可以使用它。

如果 `.venv` 或关键依赖缺失，启动器会提示先运行 `setup_amazon_review_analyzer.command`，而不是显示 Python traceback。

### Terminal 手动启动

```bash
cd /path/to/amazon_review_analyzer
source .venv/bin/activate
streamlit run app.py
```

浏览器通常会自动打开 [http://localhost:8501](http://localhost:8501)。按启动 Terminal 中的 `Ctrl+C` 可停止这次 Dashboard，不会删除数据或停止 Ollama。

## First Run / 首次运行

第一次使用时，成功打开浏览器后会显示单屏 onboarding。选择 **English** 或 **简体中文**，查看本地环境状态，然后进入 Demo Dataset、Saved Analyses 或当前可用的输入入口。只有在你继续进入应用后才会保存完成状态；创建或安装 `.app` 不会预先把 onboarding 标记为完成。以后重新启动会直接进入 Dashboard，Quick Start 仍可在文档中查看。

## Limited Mode

如果 Ollama 未安装、未运行或 `qwen3:8b` 缺失，应用仍可进入 **Limited Mode**。以下功能继续可用：

- Demo Dataset；
- Saved Analyses；
- 已保存结果的 Competitor Comparison；
- 已生成报告与文档。

需要新模型推理的 Paste、Upload 和 Amazon 新分析会被安全阻止。已有结果不会受影响；恢复 Ollama 后刷新 System Status，即可继续或 Resume。

Dashboard 会显示：**本地 AI 当前不可用 — 已有结果与查看功能仍可使用。** 这是预期的降级状态，不代表应用启动失败。

## Amazon Quick Scan（可选）

只有准备使用 Amazon Quick Scan 时，才安装浏览器 runtime：

```bash
source .venv/bin/activate
playwright install chromium
```

Quick Scan 只读取当前普通浏览器会话可以访问的评论，不绕过 CAPTCHA、登录或其他访问控制。访问受限时请使用 **Paste Reviews** 或 **Upload Dataset**。

## 常见问题

### Ollama 未安装

从 Ollama 官方网站安装。Setup 不会静默安装系统软件。Demo 和已有结果仍可使用。

### Ollama 已安装但未运行

打开 Ollama 应用，等待本地服务启动，然后在 Dashboard 的 **System Status** 中刷新，或重新运行 Health Check。

### `qwen3:8b` 缺失

```bash
ollama pull qwen3:8b
```

下载完成后刷新 System Status。不要删除其他本地模型。

### Python 环境或依赖缺失

重新双击项目根目录中的 `setup_amazon_review_analyzer.command`。Setup 会复用现有有效 `.venv`，不会删除用户文件或覆盖 `.env`。

### Port 8501 已占用

先打开 [http://localhost:8501](http://localhost:8501) 检查是否已经是本项目。桌面启动器会复用本项目实例；如果属于其他应用，请正常关闭该应用，不要自动 kill 未知进程。

### 原生 App 没有打开

确认源码项目中仍有 `main.py`、`app.py`、`environment_health.py` 与 `.venv`。如果项目已移动，请从项目内的 App 启动，或重新安装 App，以刷新私有的项目位置偏好；启动器不会扫描整台 Mac。开发版未签名时，只使用 Finder 的一次性 **右键 → 打开**，不要在 Terminal 中绕过 Gatekeeper。

### 分析被中断

生产运行会持续保存 checkpoint。重新启动应用并选择 Resume；不要手工修改 checkpoint。

### 如何继续上一次分析

恢复 Ollama 后，在同一数据集的当前分析页面选择 **Resume**。已有 `success` 与 `skipped` 评论不会重复分析；**Retry Failed Reviews** 只处理 `failed` 行。

## 数据与隐私

默认分析在本机 Ollama 中完成。评论、checkpoint、Saved Analyses 和报告保存在本机项目工作区。Dashboard 启动、Demo 浏览和 Health Check 不会自动访问 Amazon，也不会调用 OpenAI。

默认配置不依赖 OpenAI，也不需要 OpenAI API key。Amazon Quick Scan 是唯一会访问 Amazon 的输入路径；可用性取决于当前网站、网络和浏览器 session。Amazon 可能要求登录或人工验证、限制页面或只返回部分评论。系统绝不绕过 CAPTCHA、登录、反爬控制或 session 限制；Quick Scan 不可用时，请使用 **Paste Reviews** 或 **Upload Dataset**。

正式分析保持 **1 worker** 串行运行。速度取决于 Mac 硬件、评论长度与模型冷启动。

