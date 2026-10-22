#!/bin/zsh

# Amazon Review Analyzer first-time setup for macOS.
# This script intentionally does not install Ollama, download qwen3:8b,
# access Amazon, or run review inference.

emulate -L zsh
setopt PIPE_FAIL

SCRIPT_PATH="${0:A}"
SCRIPT_DIR="${SCRIPT_PATH:h}"
PROJECT_DIR="$SCRIPT_DIR"
if [[ ! -f "$PROJECT_DIR/requirements.txt" ]]; then
  CANDIDATE_PROJECT="${SCRIPT_DIR:h:h}"
  if [[ -f "$CANDIDATE_PROJECT/requirements.txt" ]]; then
    PROJECT_DIR="$CANDIDATE_PROJECT"
  else
    PROJECT_DIR="$HOME/Documents/Projects/amazon_review_analyzer"
  fi
fi
VENV_DIR="$PROJECT_DIR/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
REQUIREMENTS_FILE="$PROJECT_DIR/requirements.txt"
MINIMUM_PYTHON="3.11"
MODEL_NAME="qwen3:8b"

PYTHON_STATUS="Not checked"
ENVIRONMENT_STATUS="Not checked"
DEPENDENCIES_STATUS="Not checked"
OLLAMA_STATUS="Not checked"
MODEL_STATUS="Not checked"
FINAL_STATUS="ACTION REQUIRED"
EXIT_STATUS=1

print_header() {
  printf '\nAmazon Review Analyzer Setup\n'
  printf '%s\n\n' '----------------------------'
}

print_summary() {
  printf '\nSetup Summary\n'
  printf '%s\n' '-------------'
  printf '%-14s %s\n' "Python" "$PYTHON_STATUS"
  printf '%-14s %s\n' "Environment" "$ENVIRONMENT_STATUS"
  printf '%-14s %s\n' "Dependencies" "$DEPENDENCIES_STATUS"
  printf '%-14s %s\n' "Ollama" "$OLLAMA_STATUS"
  printf '%-14s %s\n' "$MODEL_NAME" "$MODEL_STATUS"
  printf '\nStatus: %s\n' "$FINAL_STATUS"
}

pause_before_close() {
  if [[ -t 0 ]]; then
    printf '\nPress Return to close this window.\n'
    read -r REPLY
  fi
}

finish_setup() {
  print_summary
  pause_before_close
  exit "$EXIT_STATUS"
}

print_header

if [[ ! -d "$PROJECT_DIR" || ! -f "$REQUIREMENTS_FILE" ]]; then
  printf 'The project files required for setup were not found.\n'
  printf 'Keep the source project in its recommended location or run Setup from the project folder.\n'
  FINAL_STATUS="FATAL"
  EXIT_STATUS=2
  finish_setup
fi

cd -- "$PROJECT_DIR" || {
  printf 'The project folder could not be opened.\n'
  FINAL_STATUS="FATAL"
  EXIT_STATUS=2
  finish_setup
}

printf 'Checking Python...\n'
PYTHON_COMMAND="$(command -v python3 2>/dev/null)"
if [[ -z "$PYTHON_COMMAND" ]]; then
  PYTHON_STATUS="Missing"
  printf 'Python 3 is not installed or cannot be found.\n'
  printf 'Install Python %s or later, then run Setup again.\n' "$MINIMUM_PYTHON"
  FINAL_STATUS="FATAL"
  EXIT_STATUS=2
  finish_setup
fi

PYTHON_VERSION="$("$PYTHON_COMMAND" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))' 2>/dev/null)"
if ! "$PYTHON_COMMAND" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; then
  PYTHON_STATUS="Unsupported ($PYTHON_VERSION)"
  printf 'Python version is not supported.\n'
  printf 'Required: %s or later\n' "$MINIMUM_PYTHON"
  printf 'Detected: %s\n' "${PYTHON_VERSION:-Unknown}"
  FINAL_STATUS="FATAL"
  EXIT_STATUS=2
  finish_setup
fi
PYTHON_STATUS="Ready ($PYTHON_VERSION)"

printf 'Preparing the project environment...\n'
if [[ -e "$VENV_DIR" && ! -x "$VENV_PYTHON" ]]; then
  ENVIRONMENT_STATUS="Action Required"
  printf 'An incomplete virtual environment already exists.\n'
  printf 'Setup will not delete or replace it automatically.\n'
  printf 'Move the existing .venv aside after reviewing it, then run Setup again.\n'
  FINAL_STATUS="FATAL"
  EXIT_STATUS=2
  finish_setup
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
  if ! "$PYTHON_COMMAND" -m venv "$VENV_DIR"; then
    ENVIRONMENT_STATUS="Failed"
    printf 'The Python virtual environment could not be created.\n'
    printf 'Review the Python installation, then run Setup again.\n'
    FINAL_STATUS="FATAL"
    EXIT_STATUS=2
    finish_setup
  fi
fi
ENVIRONMENT_STATUS="Ready"

printf 'Installing project dependencies from requirements.txt...\n'
if "$VENV_PYTHON" -m pip install -r "$REQUIREMENTS_FILE"; then
  DEPENDENCIES_STATUS="Ready"
else
  DEPENDENCIES_STATUS="Action Required"
  printf '\nSome required Python packages could not be installed.\n'
  printf 'Check the internet connection and run Setup again.\n'
fi

# Create the local configuration only when it does not already exist.
# Existing user configuration is never overwritten.
if [[ ! -e "$PROJECT_DIR/.env" && -f "$PROJECT_DIR/.env.example" ]]; then
  if ! /bin/cp -n "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"; then
    printf 'The optional local configuration file could not be created.\n'
  fi
fi

printf 'Checking Ollama...\n'
OLLAMA_COMMAND="$(command -v ollama 2>/dev/null)"
if [[ -z "$OLLAMA_COMMAND" ]]; then
  OLLAMA_STATUS="Not installed"
  MODEL_STATUS="Unavailable"
  printf 'Ollama is not installed or cannot be found.\n'
  printf 'Install Ollama from the official Ollama website, then run Setup again.\n'
else
  OLLAMA_VERSION="$("$OLLAMA_COMMAND" --version 2>/dev/null | head -n 1)"
  [[ -n "$OLLAMA_VERSION" ]] && OLLAMA_STATUS="Installed" || OLLAMA_STATUS="Installed (version unavailable)"

  OLLAMA_BASE_URL="${OLLAMA_HOST:-}"
  if [[ -z "$OLLAMA_BASE_URL" && -x "$VENV_PYTHON" && -f "$PROJECT_DIR/.env" ]]; then
    OLLAMA_BASE_URL="$($VENV_PYTHON -c 'import sys; from dotenv import dotenv_values; print(dotenv_values(sys.argv[1]).get("OLLAMA_HOST") or "")' "$PROJECT_DIR/.env" 2>/dev/null)"
  fi
  [[ -z "$OLLAMA_BASE_URL" ]] && OLLAMA_BASE_URL="http://localhost:11434"
  [[ "$OLLAMA_BASE_URL" != http://* && "$OLLAMA_BASE_URL" != https://* ]] && OLLAMA_BASE_URL="http://$OLLAMA_BASE_URL"
  OLLAMA_BASE_URL="${OLLAMA_BASE_URL%/}"

  TAGS_JSON="$($VENV_PYTHON -c 'import sys, urllib.request; print(urllib.request.urlopen(sys.argv[1] + "/api/tags", timeout=2).read().decode("utf-8"))' "$OLLAMA_BASE_URL" 2>/dev/null)"
  if [[ -z "$TAGS_JSON" ]]; then
    OLLAMA_STATUS="Installed, not running"
    MODEL_STATUS="Not checked"
    printf 'Ollama is installed but not running.\n'
    printf 'Start Ollama, then run Setup or the desktop launcher again.\n'
  elif print -r -- "$TAGS_JSON" | "$VENV_PYTHON" -c 'import json, sys; data=json.load(sys.stdin); names={str(item.get("name") or item.get("model") or "") for item in data.get("models", [])}; raise SystemExit(0 if "qwen3:8b" in names else 1)' >/dev/null 2>&1; then
    OLLAMA_STATUS="Ready"
    MODEL_STATUS="Ready"
  else
    OLLAMA_STATUS="Ready"
    MODEL_STATUS="Missing"
    printf '%s is not installed.\n' "$MODEL_NAME"
    printf 'The model requires several GB of local storage.\n'
    printf 'Install it when ready with:\n'
    printf '  ollama pull %s\n' "$MODEL_NAME"
  fi
fi

if [[ "$PYTHON_STATUS" == Ready* && "$ENVIRONMENT_STATUS" == "Ready" && "$DEPENDENCIES_STATUS" == "Ready" && "$OLLAMA_STATUS" == "Ready" && "$MODEL_STATUS" == "Ready" ]]; then
  FINAL_STATUS="READY"
  EXIT_STATUS=0
else
  FINAL_STATUS="ACTION REQUIRED"
  EXIT_STATUS=1
  printf '\nDemo Dataset, saved analyses, and existing reports can still be used without local AI.\n'
  printf 'New review analysis requires Ollama and %s.\n' "$MODEL_NAME"
fi

printf '\nSetup does not install Ollama, download a model, access Amazon, or run AI inference.\n'
finish_setup
