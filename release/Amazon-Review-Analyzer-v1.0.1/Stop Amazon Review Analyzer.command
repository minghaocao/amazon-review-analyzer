#!/bin/zsh

# Stops only the Streamlit PID recorded and verified by mac_launcher.py.

SCRIPT_PATH="${0:A}"
SCRIPT_DIR="${SCRIPT_PATH:h}"
PROJECT_DIR="${SCRIPT_DIR:h}"

if [[ ! -f "$PROJECT_DIR/mac_launcher.py" ]]; then
    CANDIDATE_PROJECT="${SCRIPT_DIR:h:h}"
    if [[ -f "$CANDIDATE_PROJECT/mac_launcher.py" ]]; then
        PROJECT_DIR="$CANDIDATE_PROJECT"
    else
        PROJECT_DIR="$HOME/Documents/Projects/amazon_review_analyzer"
    fi
fi

if [[ ! -x "$PROJECT_DIR/.venv/bin/python" || ! -f "$PROJECT_DIR/mac_launcher.py" ]]; then
    printf "Amazon Review Analyzer launcher files were not found.\n"
    printf "Amazon Review Analyzer 启动文件未找到。\n"
    [[ -t 0 ]] && read -r "?Press Return to close / 按回车键关闭。"
    exit 1
fi

"$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/mac_launcher.py" \
    --stop-dashboard \
    --project-root "$PROJECT_DIR"
STATUS=$?

if [[ "$STATUS" -eq 0 ]]; then
    printf "Amazon Review Analyzer stopped.\n"
    printf "Amazon Review Analyzer 已停止。\n"
fi

[[ -t 0 ]] && read -r "?Press Return to close / 按回车键关闭。"
exit "$STATUS"
