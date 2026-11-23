#!/usr/bin/env bash
# One-shot setup: system packages -> libtorch (into <repo>/libtorch) -> build -> tests.
#
#   scripts/setup.sh                     # auto: CUDA libtorch if an NVIDIA GPU is visible, else CPU
#   scripts/setup.sh --variant cu121     # force a libtorch variant: cpu, cu118, cu121, cu124
#   scripts/setup.sh --no-install        # don't apt-get install missing packages
#   scripts/setup.sh --no-tests          # skip running the unit tests
#   scripts/setup.sh --debug             # Debug build instead of Release
#   scripts/setup.sh --force-libtorch    # re-download libtorch even if ./libtorch exists
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

VARIANT="auto"
INSTALL=1
TESTS=1
BUILD_TYPE="Release"
FORCE_LIBTORCH=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant) VARIANT="$2"; shift 2 ;;
    --no-install) INSTALL=0; shift ;;
    --no-tests) TESTS=0; shift ;;
    --debug) BUILD_TYPE="Debug"; shift ;;
    --force-libtorch) FORCE_LIBTORCH="--force"; shift ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 1 ;;
  esac
done

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*"; }

# ---- 1. System packages -------------------------------------------------------
step "Checking system packages"
PACKAGES=(build-essential cmake ninja-build libgmp-dev curl unzip)
MISSING=()
for p in "${PACKAGES[@]}"; do
  dpkg -s "$p" >/dev/null 2>&1 || MISSING+=("$p")
done
if [[ ${#MISSING[@]} -eq 0 ]]; then
  echo "all present: ${PACKAGES[*]}"
elif [[ $INSTALL -eq 1 ]]; then
  echo "installing: ${MISSING[*]}"
  sudo apt-get update
  sudo apt-get install -y "${MISSING[@]}"
else
  echo "missing: ${MISSING[*]}" >&2
  echo "install them (sudo apt-get install ${MISSING[*]}) or drop --no-install" >&2
  exit 1
fi

# ---- 2. GPU / CUDA ------------------------------------------------------------
step "Detecting GPU"
HAS_GPU=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  HAS_GPU=1
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
else
  echo "no NVIDIA GPU visible (nvidia-smi unavailable)"
fi
if [[ "$VARIANT" == "auto" ]]; then
  if [[ $HAS_GPU -eq 1 ]]; then VARIANT="cu124"; else VARIANT="cpu"; fi
fi
echo "libtorch variant: $VARIANT"
if [[ "$VARIANT" != "cpu" ]]; then
  [[ $HAS_GPU -eq 1 ]] || warn "CUDA libtorch requested but no NVIDIA GPU/driver is visible"
  command -v nvcc >/dev/null 2>&1 \
    || warn "nvcc not found: libtorch's CMake config needs a CUDA toolkit (e.g. sudo apt install nvidia-cuda-toolkit)"
fi

# ---- 3. libtorch ----------------------------------------------------------------
step "Fetching libtorch into $REPO_DIR/libtorch"
if [[ -d libtorch ]] && [[ -z "$FORCE_LIBTORCH" ]]; then
  INSTALLED="$(cat libtorch/build-version 2>/dev/null || echo unknown)"
  echo "already present: $INSTALLED"
  if [[ "$VARIANT" == "cpu" && "$INSTALLED" != *"+cpu" ]] || [[ "$VARIANT" != "cpu" && "$INSTALLED" != *"+$VARIANT" ]]; then
    warn "installed libtorch ($INSTALLED) differs from the requested variant ($VARIANT); use --force-libtorch to replace it"
  fi
else
  "$REPO_DIR/scripts/fetch_libtorch.sh" "$VARIANT" ${FORCE_LIBTORCH:+--force}
fi

# ---- 4. Build -------------------------------------------------------------------
step "Configuring ($BUILD_TYPE)"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
step "Building"
cmake --build build

# ---- 5. Tests -------------------------------------------------------------------
if [[ $TESTS -eq 1 ]]; then
  step "Running tests"
  ctest --test-dir build --output-on-failure
fi

step "Setup complete"
for exe in polypack_train polypack_evaluate; do
  [[ -x "build/$exe" ]] || { warn "build/$exe was not built (libtorch not found?)"; exit 1; }
done
cat <<EOF
Next:
  ./build/polypack_train --iterations 2000      # checkpoints go to models/ (committed to git)
  ./build/polypack_evaluate --model models/policy.pt --episodes 64 --svg-dir eval_svg
EOF
