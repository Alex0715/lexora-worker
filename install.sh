#!/usr/bin/env bash
set -e

REPO_URL="https://github.com/Alex0715/lexora-worker"
ORCHESTRATOR_URL="${LEXORA_ORCHESTRATOR_URL:-https://api.lexora.network}"

# ── colours ───────────────────────────────────────────────────────────────────
BOLD="\033[1m"
GREEN="\033[0;32m"
CYAN="\033[0;36m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
RESET="\033[0m"

info()    { echo -e "${CYAN}${BOLD}▶${RESET} $*"; }
success() { echo -e "${GREEN}${BOLD}✓${RESET} $*"; }
warn()    { echo -e "${YELLOW}${BOLD}⚠${RESET} $*"; }
error()   { echo -e "${RED}${BOLD}✗${RESET} $*"; exit 1; }

# ── header ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║       Lexora Worker — Installer           ║${RESET}"
echo -e "${BOLD}║  Distributed AI Compute Node Setup       ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════╝${RESET}"
echo ""

# ── detect OS ─────────────────────────────────────────────────────────────────
OS="$(uname -s)"
ARCH="$(uname -m)"
info "Detected: $OS $ARCH"

# ── check Python ──────────────────────────────────────────────────────────────
PYTHON=""
for cmd in python3.12 python3.11 python3.10 python3.9 python3; do
    if command -v "$cmd" &>/dev/null; then
        VER=$("$cmd" -c "import sys; print(sys.version_info[:2])")
        if "$cmd" -c "import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)" 2>/dev/null; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    error "Python 3.9+ is required. Install from https://python.org and re-run this script."
fi
success "Python found: $("$PYTHON" --version)"

# ── detect GPU ────────────────────────────────────────────────────────────────
EXTRAS="mac"
if [ "$OS" = "Darwin" ]; then
    success "Apple Silicon detected — using MLX backend"
    EXTRAS="mac"
elif command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown GPU")
    success "NVIDIA GPU detected: $GPU_NAME — using vLLM backend"
    EXTRAS="inference"
else
    warn "No GPU detected — using CPU/HuggingFace backend (slower)"
    EXTRAS=""
fi

# ── install dir ───────────────────────────────────────────────────────────────
INSTALL_DIR="$HOME/.lexora-worker"
mkdir -p "$INSTALL_DIR"
info "Installing to $INSTALL_DIR"

# ── create venv ───────────────────────────────────────────────────────────────
VENV_DIR="$INSTALL_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    info "Creating virtual environment..."
    "$PYTHON" -m venv "$VENV_DIR"
    success "Virtual environment created"
else
    info "Using existing virtual environment"
fi

PIP="$VENV_DIR/bin/pip"
WORKER_BIN="$VENV_DIR/bin/lexora-worker"

# ── install worker package ────────────────────────────────────────────────────
info "Installing lexora-worker..."
"$PIP" install --quiet --upgrade pip

# Try to install from the repo's worker directory if cloned locally,
# otherwise fall back to GitHub archive
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/worker/pyproject.toml" ]; then
    info "Installing from local source..."
    if [ -n "$EXTRAS" ]; then
        "$PIP" install --quiet -e "$SCRIPT_DIR/worker[$EXTRAS]"
    else
        "$PIP" install --quiet -e "$SCRIPT_DIR/worker"
    fi
else
    info "Downloading from GitHub..."
    TMP_DIR=$(mktemp -d)
    curl -sSL "$REPO_URL/archive/refs/heads/main.tar.gz" | tar -xz -C "$TMP_DIR" --strip-components=1
    # In the public worker repo pyproject.toml is at the root, not in worker/
    if [ -n "$EXTRAS" ]; then
        "$PIP" install --quiet -e "$TMP_DIR[$EXTRAS]"
    else
        "$PIP" install --quiet -e "$TMP_DIR"
    fi
fi

success "lexora-worker installed"

# ── add to PATH ───────────────────────────────────────────────────────────────
SHELL_RC=""
if [ -n "$ZSH_VERSION" ] || [ "$SHELL" = "/bin/zsh" ]; then
    SHELL_RC="$HOME/.zshrc"
elif [ -f "$HOME/.bashrc" ]; then
    SHELL_RC="$HOME/.bashrc"
elif [ -f "$HOME/.bash_profile" ]; then
    SHELL_RC="$HOME/.bash_profile"
fi

BIN_LINE="export PATH=\"$VENV_DIR/bin:\$PATH\""
if [ -n "$SHELL_RC" ] && ! grep -q "lexora-worker" "$SHELL_RC" 2>/dev/null; then
    echo "" >> "$SHELL_RC"
    echo "# Lexora Worker" >> "$SHELL_RC"
    echo "$BIN_LINE" >> "$SHELL_RC"
    success "Added to PATH in $SHELL_RC"
fi

export PATH="$VENV_DIR/bin:$PATH"

# ── run setup wizard ──────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}Installation complete! Starting setup wizard...${RESET}"
echo ""

"$WORKER_BIN" setup
