#!/bin/zsh

# Optional local install of the thin launcher app. This script never uses sudo,
# installs dependencies, downloads a model, or changes macOS security settings.

SCRIPT_PATH="${0:A}"
SCRIPT_DIR="${SCRIPT_PATH:h}"
PROJECT_DIR="$SCRIPT_DIR"
SOURCE_APP="$SCRIPT_DIR/dist/Amazon Review Analyzer.app"
DEST_APP="/Applications/Amazon Review Analyzer.app"

# The helper may be used from the project root or from the v1.0.1 companion
# release package.  Candidate discovery is intentionally bounded; it never
# scans the user's filesystem.
if [[ -d "$SCRIPT_DIR/Amazon Review Analyzer.app" ]]; then
    SOURCE_APP="$SCRIPT_DIR/Amazon Review Analyzer.app"
fi
if [[ ! -f "$PROJECT_DIR/mac_launcher.py" ]]; then
    CANDIDATE_PROJECT="${SCRIPT_DIR:h:h}"
    if [[ -f "$CANDIDATE_PROJECT/mac_launcher.py" ]]; then
        PROJECT_DIR="$CANDIDATE_PROJECT"
    else
        PROJECT_DIR="$HOME/Documents/Projects/amazon_review_analyzer"
    fi
fi
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"

pause_before_exit() {
    if [[ -t 0 ]]; then
        printf "\nPress Return to close / 按回车键关闭。\n"
        read -r REPLY
    fi
}

if [[ ! -d "$SOURCE_APP" \
    || ! -x "$SOURCE_APP/Contents/MacOS/Amazon Review Analyzer" \
    || ! -f "$SOURCE_APP/Contents/Info.plist" ]]; then
    printf "The native application has not been built or is incomplete.\n"
    printf "原生应用尚未构建或文件不完整。\n"
    pause_before_exit
    exit 1
fi

BUNDLE_ID="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$SOURCE_APP/Contents/Info.plist" 2>/dev/null)"
if [[ "$BUNDLE_ID" != "com.caominghao.amazonreviewanalyzer" ]]; then
    printf "The application bundle could not be verified.\n"
    printf "无法验证应用程序包。\n"
    pause_before_exit
    exit 1
fi

if [[ ! -x "$VENV_PYTHON" || ! -f "$PROJECT_DIR/mac_launcher.py" ]]; then
    printf "The source project and Python environment were not found.\n"
    printf "未找到源码项目或 Python 环境。\n"
    printf "This lightweight app requires the local project; run Setup there first.\n"
    printf "此轻量启动器依赖本地项目；请先在项目目录中运行 Setup。\n"
    pause_before_exit
    exit 1
fi

if [[ -e "$DEST_APP" ]]; then
    printf "An existing Amazon Review Analyzer.app is already in /Applications.\n"
    printf "/Applications 中已存在 Amazon Review Analyzer.app。\n"
    printf "Remove or move it manually if you want to replace it.\n"
    printf "如需替换，请先手动移动或删除旧版本。\n"
    pause_before_exit
    exit 1
fi

if [[ ! -w "/Applications" ]]; then
    printf "This account cannot write to /Applications without elevated access.\n"
    printf "当前账户无法在不提升权限的情况下写入 /Applications。\n"
    printf "Drag the app there manually if desired. No system setting was changed.\n"
    printf "如有需要，请手动拖入；本程序未修改任何系统设置。\n"
    pause_before_exit
    exit 1
fi

"$VENV_PYTHON" "$PROJECT_DIR/mac_launcher.py" \
    --remember-project \
    --project-root "$PROJECT_DIR" >/dev/null 2>&1 || {
        printf "The private project location could not be saved.\n"
        printf "无法保存私有项目位置。\n"
        pause_before_exit
        exit 1
    }

if /usr/bin/ditto "$SOURCE_APP" "$DEST_APP"; then
    printf "Amazon Review Analyzer.app was copied to /Applications.\n"
    printf "Amazon Review Analyzer.app 已复制到 /Applications。\n"
    printf "This development build is unsigned and not notarized.\n"
    printf "此开发版本未签名，也未公证。\n"
    pause_before_exit
    exit 0
fi

printf "The app could not be copied. Drag it to /Applications manually.\n"
printf "无法复制 App，请手动拖入 /Applications。\n"
pause_before_exit
exit 1
