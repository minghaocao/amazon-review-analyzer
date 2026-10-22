#!/bin/zsh

# Thin Finder-safe bootstrap for Amazon Review Analyzer.app.
# Environment checks and Streamlit orchestration live in mac_launcher.py.

emulate -L zsh
setopt NO_UNSET

EXECUTABLE_PATH="${0:A}"
APP_BUNDLE="${EXECUTABLE_PATH:h:h:h}"
RESOURCES_DIR="$APP_BUNDLE/Contents/Resources"
LAUNCHER_HELPER="$RESOURCES_DIR/mac_launcher.py"
RELATIVE_PROJECT_ROOT="${APP_BUNDLE:h:h}"

show_bootstrap_error() {
  /usr/bin/osascript \
    -e 'on run argv' \
    -e 'display dialog (item 1 of argv) with title "Amazon Review Analyzer" buttons {"OK"} default button "OK" with icon stop' \
    -e 'end run' \
    "$1" >/dev/null 2>&1 || true
}

if [[ ! -f "$LAUNCHER_HELPER" ]]; then
  show_bootstrap_error $'Amazon Review Analyzer could not locate its application files.\n\nAmazon Review Analyzer 无法找到应用程序文件。'
  exit 1
fi

# Prefer the project environment when the app remains in project/dist.  The
# remaining fixed candidates cover normal macOS Python installations without
# changing PATH or a shell profile.
PYTHON_BIN=""
for candidate in \
  "$RELATIVE_PROJECT_ROOT/.venv/bin/python" \
  "$HOME/Documents/Projects/amazon_review_analyzer/.venv/bin/python" \
  "/usr/local/bin/python3" \
  "/opt/homebrew/bin/python3" \
  "/Library/Frameworks/Python.framework/Versions/Current/bin/python3" \
  "/usr/bin/python3"
do
  if [[ -x "$candidate" ]]; then
    PYTHON_BIN="$candidate"
    break
  fi
done

if [[ -z "$PYTHON_BIN" ]]; then
  show_bootstrap_error $'Python environment is not ready. Run Setup to prepare Amazon Review Analyzer.\n\nPython 环境尚未准备完成。请运行 Setup 准备 Amazon Review Analyzer。'
  exit 1
fi

# Finder launches this executable without Terminal.  The helper deliberately
# owns all project discovery, health, dialog, process, log and browser logic.
exec "$PYTHON_BIN" "$LAUNCHER_HELPER" \
  --native \
  --app-bundle "$APP_BUNDLE" \
  </dev/null >/dev/null 2>&1

