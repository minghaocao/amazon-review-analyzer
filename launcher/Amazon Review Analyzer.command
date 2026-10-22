#!/bin/zsh

# Terminal-compatible launcher. Native and Terminal entry points share the
# same Python orchestration; the default dashboard port remains 8501.

SCRIPT_PATH="${0:A}"
SCRIPT_DIR="${SCRIPT_PATH:h}"
PROJECT_DIR=""

if [[ -f "$SCRIPT_DIR/../mac_launcher.py" ]]; then
    PROJECT_DIR="${SCRIPT_DIR:h}"
elif [[ -f "$HOME/Documents/Projects/amazon_review_analyzer/mac_launcher.py" ]]; then
    PROJECT_DIR="$HOME/Documents/Projects/amazon_review_analyzer"
fi

pause_before_exit() {
    if [[ -t 0 ]]; then
        printf "\nPress any key to close this window."
        read -r -k 1
        printf "\n"
    fi
}

printf '\033]0;Amazon Review Analyzer\007'
printf "Amazon Review Analyzer\n"
printf "======================\n\n"

if [[ -z "$PROJECT_DIR" || ! -d "$PROJECT_DIR" ]]; then
    printf "Amazon Review Analyzer project folder was not found.\n"
    pause_before_exit
    exit 1
fi

VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
    printf "Python environment is not ready.\n\n"
    printf "Run setup_amazon_review_analyzer.command, then try again.\n"
    pause_before_exit
    exit 1
fi

if [[ ! -f "$PROJECT_DIR/setup_amazon_review_analyzer.command" ]]; then
    printf "Setup files are unavailable. Keep the launcher with the project folder.\n"
    pause_before_exit
    exit 1
fi

cd -- "$PROJECT_DIR" || exit 1
exec "$VENV_PYTHON" "$PROJECT_DIR/mac_launcher.py" \
    --foreground \
    --project-root "$PROJECT_DIR"
