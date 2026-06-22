#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$BASE_DIR"

echo "BASE_DIR: $BASE_DIR"

PYTORCH_CUDA_INDEX="${PYTORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu124}"
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"

if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y \
    build-essential \
    ca-certificates \
    git \
    bzip2 \
    unzip \
    wget \
    python3 \
    python3-pip \
    python3-venv
fi

python_minor_version() {
  "$1" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
}

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_VERSION="$(python_minor_version "$PYTHON_BIN")"
  if [[ "$PYTHON_VERSION" != "3.11" ]]; then
    echo "PYTHON_BIN points to Python $PYTHON_VERSION, but this project needs Python 3.11 for torch==$TORCH_VERSION."
    exit 1
  fi
elif command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3.11)"
else
  MINIFORGE_DIR="${MINIFORGE_DIR:-$HOME/.tabstruct-miniforge}"
  CONDA_PY311_ENV="${CONDA_PY311_ENV:-$HOME/.tabstruct-python311}"

  echo "python3.11 was not found in apt packages. This is normal on some Ubuntu/WSL releases."
  echo "Installing a local Miniforge Python 3.11 environment under: $CONDA_PY311_ENV"
  echo "This can take a few minutes on the first run."
  echo "Miniforge is installed under your HOME directory to avoid spaces in the repo path."

  if [[ ! -f "$MINIFORGE_DIR/etc/profile.d/conda.sh" ]]; then
    wget --progress=bar:force:noscroll \
      https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
      -O /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p "$MINIFORGE_DIR"
  fi

  set +u
  source "$MINIFORGE_DIR/etc/profile.d/conda.sh"
  set -u

  if [[ ! -x "$CONDA_PY311_ENV/bin/python" ]]; then
    conda create -y -p "$CONDA_PY311_ENV" python=3.11 pip
  fi

  PYTHON_BIN="$CONDA_PY311_ENV/bin/python"
fi

PYTHON_VERSION="$(python_minor_version "$PYTHON_BIN")"
echo "Using Python: $PYTHON_BIN ($PYTHON_VERSION)"

if [[ -x .venv/bin/python ]]; then
  VENV_VERSION="$(python_minor_version .venv/bin/python || true)"
  if [[ "$VENV_VERSION" != "3.11" ]]; then
    echo "Existing .venv uses Python $VENV_VERSION. Remove it and rerun setup:"
    echo "  rm -rf .venv"
    echo "  bash script/setup_wsl.sh"
    exit 1
  fi
else
  "$PYTHON_BIN" -m venv .venv
fi

set +u
source .venv/bin/activate
set -u

python -m pip install --upgrade pip setuptools wheel
python -m pip install "torch==$TORCH_VERSION" --index-url "$PYTORCH_CUDA_INDEX"
python -m pip install -r requirements.txt
python -m pip check
python -m compileall -q tabstruct run.py

echo "WSL setup complete. Activate with: source .venv/bin/activate"
