#!/usr/bin/env bash
# Downloads libtorch and unzips it into <repo>/libtorch, where CMake finds it
# automatically. The directory is gitignored (it is ~2 GB).
#
#   scripts/fetch_libtorch.sh            # CUDA 12.4 build if nvidia-smi works, else CPU
#   scripts/fetch_libtorch.sh cu124      # or: cu121, cu118, cpu
#   scripts/fetch_libtorch.sh cu124 --force   # replace an existing ./libtorch
set -euo pipefail

VERSION="2.5.1"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$REPO_DIR/libtorch"

VARIANT="${1:-auto}"
FORCE="${2:-}"
if [[ "$VARIANT" == "--force" ]]; then VARIANT="auto"; FORCE="--force"; fi
if [[ "$VARIANT" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    VARIANT="cu124"
  else
    VARIANT="cpu"
  fi
fi
case "$VARIANT" in
  cpu|cu118|cu121|cu124) ;;
  *) echo "unknown variant '$VARIANT' (use cpu, cu118, cu121 or cu124)" >&2; exit 1 ;;
esac

if [[ -d "$TARGET" ]]; then
  if [[ "$FORCE" != "--force" ]]; then
    echo "$TARGET already exists ($(cat "$TARGET/build-version" 2>/dev/null || echo unknown version))."
    echo "Re-run with --force to replace it."
    exit 0
  fi
  rm -rf "$TARGET"
fi

URL="https://download.pytorch.org/libtorch/${VARIANT}/libtorch-cxx11-abi-shared-with-deps-${VERSION}%2B${VARIANT}.zip"
ZIP="$REPO_DIR/libtorch-${VERSION}-${VARIANT}.zip"
echo "Downloading libtorch ${VERSION} (${VARIANT}) ..."
curl -fL --progress-bar -o "$ZIP" "$URL"
echo "Unzipping into $TARGET ..."
unzip -q "$ZIP" -d "$REPO_DIR"
rm -f "$ZIP"
echo "Done: $(cat "$TARGET/build-version")"
echo "Configure with: cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release"
