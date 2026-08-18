#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
PROJECT_SPEC="${PROJECT_SPEC:-.[dev]}"
CUDA_TAG="${CUDA_TAG:-auto}"
INSTALL_PYG_KERNELS="${INSTALL_PYG_KERNELS:-1}"
TORCH_SPEC="${TORCH_SPEC:-torch>=2.8,<2.9}"
TORCHVISION_SPEC="${TORCHVISION_SPEC:-torchvision>=0.23,<0.24}"
DOWNLOAD_TEDS_DATA="${DOWNLOAD_TEDS_DATA:-1}"
TEDS_GDOWN_FILE_ID="${TEDS_GDOWN_FILE_ID:-1T1oYAsdYDcdqUckd7CBzBWj9RnwGrEZg}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TEDS_DATA_DIR="${TEDS_DATA_DIR:-${REPO_ROOT}/src/data/raw}"
TEDS_DATA_PATH="${TEDS_DATA_PATH:-${TEDS_DATA_DIR}/TEDS_Discharge.csv}"

declare -a PIP_INSTALL_FLAGS=()

ts() { date '+%Y-%m-%d %H:%M:%S'; }

require_python() {
  "$PYTHON_BIN" - <<'PY'
import sys

if sys.version_info < (3, 10):
    raise SystemExit(f"Python >= 3.10 is required, got {sys.version.split()[0]}")
PY
}

refresh_pip_flags() {
  PIP_INSTALL_FLAGS=()
  if "$PYTHON_BIN" -m pip help install 2>/dev/null | grep -q -- "--break-system-packages"; then
    if "$PYTHON_BIN" - <<'PY'
import pathlib
import sysconfig

stdlib = pathlib.Path(sysconfig.get_paths().get("stdlib", ""))
raise SystemExit(0 if (stdlib / "EXTERNALLY-MANAGED").exists() else 1)
PY
    then
      PIP_INSTALL_FLAGS+=(--break-system-packages)
    fi
  fi
}

pip_install() {
  "$PYTHON_BIN" -m pip install "${PIP_INSTALL_FLAGS[@]}" "$@"
}

download_teds_data() {
  if [[ "$DOWNLOAD_TEDS_DATA" != "1" ]]; then
    echo "[$(ts)] skipping TEDS raw data download (DOWNLOAD_TEDS_DATA=${DOWNLOAD_TEDS_DATA})"
    return 0
  fi

  if [[ -s "$TEDS_DATA_PATH" ]]; then
    echo "[$(ts)] TEDS raw data already exists: ${TEDS_DATA_PATH}"
    return 0
  fi

  mkdir -p "$(dirname "$TEDS_DATA_PATH")"
  echo "[$(ts)] downloading TEDS raw data -> ${TEDS_DATA_PATH}"
  "$PYTHON_BIN" -m gdown "https://drive.google.com/uc?id=${TEDS_GDOWN_FILE_ID}" -O "$TEDS_DATA_PATH"

  if [[ ! -s "$TEDS_DATA_PATH" ]]; then
    echo "[$(ts)] ERROR: TEDS raw data download did not create a non-empty file: ${TEDS_DATA_PATH}" >&2
    return 1
  fi
  echo "[$(ts)] TEDS raw data ready: ${TEDS_DATA_PATH}"
}

detect_cuda_tag() {
  if [[ "$CUDA_TAG" != "auto" ]]; then
    printf '%s\n' "$CUDA_TAG"
    return 0
  fi

  if [[ "$(uname -s)" == "Darwin" ]]; then
    printf 'cpu\n'
    return 0
  fi

  local raw=""
  raw="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1\2/p' | head -n 1 || true)"
  if [[ -z "$raw" ]]; then
    raw="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9]*\)\.\([0-9]*\).*/\1\2/p' | head -n 1 || true)"
  fi

  case "$raw" in
    128|129) printf 'cu128\n' ;;
    126|127) printf 'cu126\n' ;;
    124|125) printf 'cu124\n' ;;
    "") printf 'cpu\n' ;;
    *) printf 'cu124\n' ;;
  esac
}

install_torch() {
  local accel_tag="$1"
  local system_name
  system_name="$(uname -s)"

  echo "[$(ts)] installing PyTorch for ${accel_tag}"
  if [[ "$system_name" == "Darwin" ]]; then
    pip_install "$TORCH_SPEC" "$TORCHVISION_SPEC"
  elif [[ "$accel_tag" == "cpu" ]]; then
    pip_install "$TORCH_SPEC" "$TORCHVISION_SPEC" --index-url "https://download.pytorch.org/whl/cpu"
  else
    pip_install "$TORCH_SPEC" "$TORCHVISION_SPEC" --index-url "https://download.pytorch.org/whl/${accel_tag}"
  fi
}

install_pyg() {
  local accel_tag="$1"
  local system_name
  system_name="$(uname -s)"

  echo "[$(ts)] installing PyG"
  pip_install "torch-geometric>=2.7,<2.8"

  if [[ "$INSTALL_PYG_KERNELS" != "1" ]]; then
    echo "[$(ts)] skipping optional PyG kernel wheels (INSTALL_PYG_KERNELS=${INSTALL_PYG_KERNELS})"
    return 0
  fi

  if [[ "$system_name" == "Darwin" ]]; then
    echo "[$(ts)] skipping optional PyG kernel wheels on macOS; PyTorch MPS uses the PyPI torch wheel"
    return 0
  fi

  local torch_ver
  torch_ver="$("$PYTHON_BIN" -c "import torch; print(torch.__version__.split('+')[0])")"

  local wheel_tag="$accel_tag"
  if [[ "$wheel_tag" == "cpu" ]]; then
    wheel_tag="cpu"
  fi

  echo "[$(ts)] installing PyG kernel wheels for torch-${torch_ver}+${wheel_tag}"
  pip_install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f "https://data.pyg.org/whl/torch-${torch_ver}+${wheel_tag}.html"
}

verify_install() {
  "$PYTHON_BIN" - <<'PY'
import optuna
import pandas
import sklearn
import torch
import torch_geometric
import xgboost

print("INSTALL_OK=1")
print(f"python_imports=ok")
print(f"torch={torch.__version__}")
print(f"torch_geometric={torch_geometric.__version__}")
print(f"optuna={optuna.__version__}")
print(f"xgboost={xgboost.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"mps_available={getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available()}")
PY
}

require_python
cd "$REPO_ROOT"
"$PYTHON_BIN" -m ensurepip --upgrade >/dev/null 2>&1 || true
"$PYTHON_BIN" -m pip --version >/dev/null
refresh_pip_flags
pip_install -U pip setuptools wheel
refresh_pip_flags

ACCEL_TAG="$(detect_cuda_tag)"
echo "[$(ts)] detected accelerator tag: ${ACCEL_TAG}"

install_torch "$ACCEL_TAG"
install_pyg "$ACCEL_TAG"

echo "[$(ts)] installing project: ${PROJECT_SPEC}"
pip_install "$PROJECT_SPEC"

download_teds_data
verify_install
echo "[$(ts)] pip environment setup complete"
