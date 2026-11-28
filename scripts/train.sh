#!/usr/bin/env bash
# Starts training in the background. Output goes to models/train.out.
#
#   scripts/train.sh                     # new run (overwrites models/policy*.pt)
#   scripts/train.sh --iterations 500    # extra options go to polypack_train
#   scripts/train.sh --resume            # continue from models/policy_latest.pt
#   scripts/train.sh --foreground        # run in this terminal instead
#
# Watch:  tail -f models/train.out
# Stop:   pkill -f polypack_train      (progress is saved every --eval-every iterations)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

RESUME=0
FOREGROUND=0
EXTRA=()
for a in "$@"; do
  case "$a" in
    --resume) RESUME=1 ;;
    --foreground) FOREGROUND=1 ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) EXTRA+=("$a") ;;
  esac
done

if [[ ! -d build ]]; then
  echo "build/ not found: run scripts/setup.sh first" >&2
  exit 1
fi
if pgrep -f build/polypack_train >/dev/null; then
  echo "polypack_train is already running (pkill -f polypack_train to stop it)" >&2
  exit 1
fi
cmake --build build   # all programs (train, evaluate, viewer); picks up pulled code, quick if nothing changed
mkdir -p models

if [[ $RESUME -eq 1 ]]; then
  if [[ ! -f models/policy_latest.pt ]]; then
    echo "models/policy_latest.pt not found: nothing to resume" >&2
    exit 1
  fi
  CMD=(./build/polypack_train --resume models/policy_latest.pt --iterations 200 "${EXTRA[@]}")
  REDIRECT=">>"
else
  # Defaults: 64-channel network, 4 open bins; later options override earlier ones.
  CMD=(./build/polypack_train --channels 64 --open-bins 4 --iterations 200 --eval-every 5 "${EXTRA[@]}")
  REDIRECT=">"
fi

echo "running: ${CMD[*]}"
if [[ $FOREGROUND -eq 1 ]]; then
  if [[ $REDIRECT == ">" ]]; then : > models/train.out; fi
  "${CMD[@]}" 2>&1 | tee -a models/train.out
  exit "${PIPESTATUS[0]}"
fi
if [[ $REDIRECT == ">" ]]; then
  nohup "${CMD[@]}" > models/train.out 2>&1 < /dev/null &
else
  nohup "${CMD[@]}" >> models/train.out 2>&1 < /dev/null &
fi
echo "started (pid $!). Watch with: tail -f models/train.out"
