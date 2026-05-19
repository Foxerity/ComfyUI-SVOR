#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# ComfyUI-SVOR setup script
# =============================================================================
# Expected layout:
#
#   ComfyUI/
#   +-- main.py
#   +-- custom_nodes/
#       +-- ComfyUI-SVOR/
#           +-- setup_comfyui.sh
#           +-- requirements.txt
#           +-- comfyui_svor/svor_workflow_ui.json
#
# This script assumes ComfyUI itself is already installed and this repository has
# already been cloned into ComfyUI/custom_nodes/.
#
# It only:
#   1. Installs ComfyUI-SVOR Python dependencies
#   2. Downloads missing SVOR model weights
#   3. Imports the default ComfyUI workflow template
#
# Usage:
#   bash setup_comfyui.sh [options]
#
# Options:
#   --skip-install       Skip pip install -r requirements.txt
#   --skip-torch         Skip the SVOR PyTorch version check/install
#   --install-flash-attn Install optional flash-attn from the SVOR README
#   --skip-weights       Skip model weight downloads
#   --models-dir DIR     Model directory, absolute or relative to this repo (default: models)
#
# Environment variables:
#   PYTHON_BIN           Python executable used for pip (default: python or python3)
#   SVOR_TORCH_PACKAGES  PyTorch stack from the SVOR README
#   SVOR_TORCH_INSTALL_CMD
#                        Full custom torch install command, overrides SVOR_TORCH_PACKAGES
#   INSTALL_FLASH_ATTN   Set to 1 to install optional flash-attn
#   FLASH_ATTN_VERSION   flash-attn version (default: 2.7.4.post1)
#   HF_TOKEN             Hugging Face token for gated model downloads
#   HF_ENDPOINT          Hugging Face mirror endpoint (default: https://huggingface.co)
#   WAN_REPO             Base model repo (default: Wan-AI/Wan2.1-VACE-1.3B)
# =============================================================================

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SVOR_ROOT="$SCRIPT_PATH"
COMFYUI_DIR="$(cd "$SVOR_ROOT/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-}"
MODELS_DIR="models"
SKIP_INSTALL=0
SKIP_TORCH=0
SKIP_WEIGHTS=0
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"
SVOR_TORCH_PACKAGES="${SVOR_TORCH_PACKAGES:-torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0}"
SVOR_TORCH_INSTALL_CMD="${SVOR_TORCH_INSTALL_CMD:-}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.7.4.post1}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-install) SKIP_INSTALL=1; shift ;;
        --skip-torch) SKIP_TORCH=1; shift ;;
        --install-flash-attn) INSTALL_FLASH_ATTN=1; shift ;;
        --skip-weights) SKIP_WEIGHTS=1; shift ;;
        --models-dir)   MODELS_DIR="$2"; shift 2 ;;
        -h|--help)
            cat <<'EOF'
ComfyUI-SVOR setup

Expected location:
  /path/to/ComfyUI/custom_nodes/ComfyUI-SVOR

Usage:
  bash setup_comfyui.sh [options]

Options:
  --skip-install       Skip pip install -r requirements.txt
  --skip-torch         Skip the SVOR PyTorch version check/install
  --install-flash-attn Install optional flash-attn from the SVOR README
  --skip-weights       Skip model weight downloads
  --models-dir DIR     Model directory, absolute or relative to this repo (default: models)

Environment:
  PYTHON_BIN           Python executable used for pip (default: python or python3)
  SVOR_TORCH_PACKAGES  PyTorch stack from the SVOR README
  SVOR_TORCH_INSTALL_CMD
                       Full custom torch install command, overrides SVOR_TORCH_PACKAGES
  INSTALL_FLASH_ATTN   Set to 1 to install optional flash-attn
  FLASH_ATTN_VERSION   flash-attn version (default: 2.7.4.post1)
  HF_TOKEN             Hugging Face token for gated model downloads
  HF_ENDPOINT          Hugging Face mirror endpoint (default: https://huggingface.co)
  WAN_REPO             Base model repo (default: Wan-AI/Wan2.1-VACE-1.3B)
EOF
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[ComfyUI-SVOR]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

find_python() {
    if [[ -n "$PYTHON_BIN" ]]; then
        command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
            error "PYTHON_BIN is set but not executable: $PYTHON_BIN"
            exit 1
        }
        echo "$PYTHON_BIN"
        return 0
    fi

    if command -v python >/dev/null 2>&1; then
        echo python
    elif command -v python3 >/dev/null 2>&1; then
        echo python3
    else
        error "python or python3 is required."
        exit 1
    fi
}

resolve_path_from_root() {
    local path="$1"
    if [[ "$path" = /* ]]; then
        echo "$path"
    else
        echo "$SVOR_ROOT/$path"
    fi
}

file_ready() {
    [[ -s "$1" ]]
}

wan_model_ready() {
    local dir="$1"
    file_ready "$dir/config.json" &&
    file_ready "$dir/diffusion_pytorch_model.safetensors" &&
    file_ready "$dir/Wan2.1_VAE.pth" &&
    file_ready "$dir/models_t5_umt5-xxl-enc-bf16.pth" &&
    file_ready "$dir/google/umt5-xxl/tokenizer.json" &&
    file_ready "$dir/google/umt5-xxl/spiece.model"
}

svor_lora_weights_ready() {
    local dir="$1"
    file_ready "$dir/remove_model_stage1.safetensors" &&
    file_ready "$dir/remove_model_stage2.safetensors"
}

torch_ready() {
    "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import importlib

expected = {
    "torch": "2.7.0",
    "torchvision": "0.22.0",
    "torchaudio": "2.7.0",
}

for name, version in expected.items():
    module = importlib.import_module(name)
    installed = getattr(module, "__version__", "").split("+", 1)[0]
    if installed != version:
        raise SystemExit(f"{name}=={installed}, expected {version}")
PY
}

install_torch_stack() {
    if [[ "$SKIP_TORCH" -eq 1 ]]; then
        info "Skipping SVOR PyTorch stack check/install (--skip-torch)."
        return 0
    fi

    if torch_ready; then
        info "SVOR PyTorch stack already present: $SVOR_TORCH_PACKAGES"
        return 0
    fi

    info "Installing SVOR PyTorch stack before other dependencies..."
    if [[ -n "$SVOR_TORCH_INSTALL_CMD" ]]; then
        info "Running custom command: $SVOR_TORCH_INSTALL_CMD"
        bash -lc "$SVOR_TORCH_INSTALL_CMD"
    else
        # Intentionally allow word splitting so users can add pip options in SVOR_TORCH_PACKAGES.
        # shellcheck disable=SC2086
        "$PYTHON_BIN" -m pip install $SVOR_TORCH_PACKAGES
    fi

    if ! torch_ready; then
        warn "Installed PyTorch stack does not exactly match the SVOR README versions."
        warn "Continuing because CUDA/index builds may include local version suffixes."
    fi
}

flash_attn_ready() {
    "$PYTHON_BIN" - "$FLASH_ATTN_VERSION" <<'PY' >/dev/null 2>&1
import importlib
import sys

expected = sys.argv[1]
module = importlib.import_module("flash_attn")
installed = getattr(module, "__version__", "")
if installed != expected:
    raise SystemExit(f"flash_attn=={installed}, expected {expected}")
PY
}

install_flash_attn() {
    if [[ "$INSTALL_FLASH_ATTN" != "1" ]]; then
        return 0
    fi

    if flash_attn_ready; then
        info "Optional flash-attn already present: flash-attn==$FLASH_ATTN_VERSION"
        return 0
    fi

    info "Installing optional flash-attn stack from the SVOR README..."
    "$PYTHON_BIN" -m pip install packaging ninja psutil
    "$PYTHON_BIN" -m pip install "flash-attn==$FLASH_ATTN_VERSION" --no-build-isolation
}

validate_layout() {
    if [[ "$(basename "$(dirname "$SVOR_ROOT")")" != "custom_nodes" ]]; then
        error "This repository must be cloned under ComfyUI/custom_nodes/."
        error "Current path: $SVOR_ROOT"
        error "Expected: /path/to/ComfyUI/custom_nodes/ComfyUI-SVOR"
        exit 1
    fi

    if [[ ! -f "$COMFYUI_DIR/main.py" ]]; then
        error "Cannot find ComfyUI main.py at: $COMFYUI_DIR/main.py"
        error "Please run this script from /path/to/ComfyUI/custom_nodes/ComfyUI-SVOR."
        exit 1
    fi

    if [[ ! -f "$SVOR_ROOT/comfyui_svor/svor_workflow_ui.json" ]]; then
        error "Workflow template not found: $SVOR_ROOT/comfyui_svor/svor_workflow_ui.json"
        exit 1
    fi
}

install_dependencies() {
    if [[ "$SKIP_INSTALL" -eq 1 ]]; then
        info "Skipping dependency installation (--skip-install)."
        return 0
    fi

    install_torch_stack

    if [[ -f "$SVOR_ROOT/requirements.txt" ]]; then
        info "Installing ComfyUI-SVOR dependencies from requirements.txt..."
        "$PYTHON_BIN" -m pip install -r "$SVOR_ROOT/requirements.txt"
        info "Dependencies installed."
    else
        warn "requirements.txt not found, skipping dependency installation."
    fi

    install_flash_attn
}

download_weights() {
    if [[ "$SKIP_WEIGHTS" -eq 1 ]]; then
        info "Skipping weight download (--skip-weights)."
        return 0
    fi

    local models_abs="$1"
    local wan_dir="$models_abs/Wan2.1-VACE-1.3B"

    mkdir -p "$models_abs"
    info "Checking model weights under: $models_abs"

    if wan_model_ready "$wan_dir"; then
        info "Wan2.1-VACE-1.3B already complete, skipping."
    else
        info "Downloading Wan2.1-VACE-1.3B..."
        if command -v huggingface-cli >/dev/null 2>&1; then
            HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}" \
            HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}" \
            HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}" \
                huggingface-cli download "${WAN_REPO:-Wan-AI/Wan2.1-VACE-1.3B}" \
                --local-dir "$wan_dir" \
                ${HF_TOKEN:+--token "$HF_TOKEN"}
        else
            warn "huggingface-cli not found. Trying git clone with Git LFS..."
            if ! command -v git >/dev/null 2>&1 || ! git lfs version >/dev/null 2>&1; then
                error "huggingface-cli or git + git-lfs is required to download Wan2.1-VACE-1.3B."
                error "Install one of them, or pre-place the model under: $wan_dir"
                exit 1
            fi
            HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
            WAN_REPO="${WAN_REPO:-Wan-AI/Wan2.1-VACE-1.3B}"
            if [[ -n "${HF_TOKEN:-}" ]]; then
                git clone "https://user:${HF_TOKEN}@${HF_ENDPOINT#https://}/${WAN_REPO}" "$wan_dir"
            else
                git clone "${HF_ENDPOINT}/${WAN_REPO}" "$wan_dir"
            fi
            (cd "$wan_dir" && git lfs pull)
        fi

        if ! wan_model_ready "$wan_dir"; then
            error "Wan2.1-VACE-1.3B is incomplete under: $wan_dir"
            error "Expected: config.json, diffusion_pytorch_model.safetensors, Wan2.1_VAE.pth,"
            error "models_t5_umt5-xxl-enc-bf16.pth, google/umt5-xxl/tokenizer.json, google/umt5-xxl/spiece.model"
            exit 1
        fi
        info "Wan2.1-VACE-1.3B ready."
    fi

    if svor_lora_weights_ready "$models_abs"; then
        info "SVOR LoRA weights already complete, skipping."
    else
        info "Downloading SVOR LoRA weights..."
        bash "$SVOR_ROOT/download_weights_wget.sh" "$models_abs"
        if ! svor_lora_weights_ready "$models_abs"; then
            error "SVOR LoRA weights are incomplete under: $models_abs"
            error "Expected: remove_model_stage1.safetensors, remove_model_stage2.safetensors"
            exit 1
        fi
        info "SVOR LoRA weights ready."
    fi
}

import_workflow() {
    local workflow_src="$SVOR_ROOT/comfyui_svor/svor_workflow_ui.json"
    local workflow_dst_dir="$COMFYUI_DIR/user/default/workflows"
    local workflow_dst="$workflow_dst_dir/svor_workflow_ui.json"

    mkdir -p "$workflow_dst_dir"
    cp -f "$workflow_src" "$workflow_dst"
    info "Workflow template imported to: $workflow_dst"
}

PYTHON_BIN="$(find_python)"
MODELS_ABS="$(resolve_path_from_root "$MODELS_DIR")"

validate_layout

info "Repository: $SVOR_ROOT"
info "ComfyUI: $COMFYUI_DIR"

install_dependencies
download_weights "$MODELS_ABS"
import_workflow

info "Setup complete. Restart ComfyUI, then load workflow: svor_workflow_ui.json"
