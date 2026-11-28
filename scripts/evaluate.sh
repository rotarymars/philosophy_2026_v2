#!/usr/bin/env bash
# Evaluates a trained model against first-fit, greedy and the offline reference.
# Uses the GPU if there is one and every CPU core. Output also goes to models/eval*.out.
#
#   scripts/evaluate.sh                  # models/policy.pt on the 8 training-eval episodes (fast, cached)
#   scripts/evaluate.sh --latest         # models/policy_latest.pt instead
#   scripts/evaluate.sh --unseen         # 8 episodes never used in training (slow first time:
#                                        #   offline reference ~2-3 h, runs in the background)
#   scripts/evaluate.sh --background     # run in the background
#   scripts/evaluate.sh --episodes 16    # extra options go to polypack_evaluate
#
# Watch a background run:  tail -f models/eval.out   (or models/eval_unseen.out)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

MODEL=models/policy.pt
UNSEEN=0
BACKGROUND=0
EXTRA=()
for a in "$@"; do
  case "$a" in
    --latest) MODEL=models/policy_latest.pt ;;
    --unseen) UNSEEN=1; BACKGROUND=1 ;;
    --background) BACKGROUND=1 ;;
    --foreground) BACKGROUND=0 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) EXTRA+=("$a") ;;
  esac
done

if [[ ! -d build ]]; then
  echo "build/ not found: run scripts/setup.sh first" >&2
  exit 1
fi
if [[ ! -f $MODEL ]]; then
  echo "$MODEL not found (git pull, or train first)" >&2
  exit 1
fi
cmake --build build --target polypack_evaluate   # picks up pulled code; quick if nothing changed

CMD=(./build/polypack_evaluate --model "$MODEL" --greedy-baseline)
OUT=models/eval.out
if [[ $UNSEEN -eq 1 ]]; then
  CMD+=(--seed-base 2000000000)
  OUT=models/eval_unseen.out
fi
CMD+=("${EXTRA[@]}")

echo "running: ${CMD[*]}"
if [[ $BACKGROUND -eq 1 ]]; then
  nohup "${CMD[@]}" > "$OUT" 2>&1 < /dev/null &
  echo "started (pid $!). Watch with: tail -f $OUT"
  echo "new offline results are saved to models/offline_cache.tsv, so the next run is fast"
else
  "${CMD[@]}" 2>&1 | tee "$OUT"
  exit "${PIPESTATUS[0]}"
fi
