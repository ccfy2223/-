#!/usr/bin/env bash
set -euo pipefail

# ====== User config ======
GPU_ID=0
INPUT_DIR="/root/autodl-tmp/processed_csv/aligned_stations"
META_PATH="/root/autodl-tmp/processed_csv/shared_timeline_metadata.json"
TEACHER_CACHE="/root/autodl-tmp/Untitled Folder/chronos_teacher_6h"
OUT_ROOT="/root/sota_runs/weight_grid_wvht"
TFT_TIME_LIMIT=21600
WEIGHTS=("0.00" "0.03" "0.05" "0.10")
# =========================

DISTILL_SCRIPT="/root/autodl-tmp/train_tft_distill.py"
ITER_SCRIPT="/root/autodl-tmp/train_tft_iterative.py"

if [[ ! -f "$DISTILL_SCRIPT" ]]; then
  echo "Missing $DISTILL_SCRIPT"
  exit 1
fi
if [[ ! -f "$ITER_SCRIPT" ]]; then
  echo "Missing $ITER_SCRIPT"
  exit 1
fi
if [[ ! -f "$TEACHER_CACHE/teacher_forecasts_train.csv" ]]; then
  echo "Missing teacher cache under: $TEACHER_CACHE"
  exit 1
fi

if ! grep -q "except TypeError" "$ITER_SCRIPT"; then
  echo "train_tft_iterative.py seems unpatched for reset_paths compatibility."
  echo "Please patch it first, then rerun."
  exit 1
fi

mkdir -p "$OUT_ROOT"
SUMMARY_CSV="$OUT_ROOT/short_grid_summary.csv"
echo "teacher_weight,val_mase,test_mase,run_dir,model_dir,status" > "$SUMMARY_CSV"

echo "==== [1/3] Short-horizon grid (distill) ===="
for W in "${WEIGHTS[@]}"; do
  TAG="${W/./}"
  RUN_DIR="$OUT_ROOT/w${TAG}"
  mkdir -p "$RUN_DIR"

  echo
  echo ">>> Running teacher_weight=$W"
  set +e
  PYTHONUNBUFFERED=1 python -u "$DISTILL_SCRIPT" \
    --gpu-id "$GPU_ID" \
    --input-dir "$INPUT_DIR" \
    --metadata-path "$META_PATH" \
    --teacher-cache-dir "$TEACHER_CACHE" \
    --output-root "$RUN_DIR" \
    --log-path "$RUN_DIR/distill_log.json" \
    --short-horizon-hours 6 \
    --context-hours 168 \
    --teacher-weight "$W" \
    --tft-time-limit "$TFT_TIME_LIMIT" \
    --verbosity 3 \
    > "$RUN_DIR/run_short.log" 2>&1
  EXIT_CODE=$?
  set -e

  if [[ $EXIT_CODE -ne 0 ]]; then
    echo "teacher_weight=$W FAILED (exit=$EXIT_CODE). See $RUN_DIR/run_short.log"
    echo "$W,NA,NA,$RUN_DIR,NA,failed" >> "$SUMMARY_CSV"
    continue
  fi

  readarray -t METRICS < <(python - "$RUN_DIR/distill_log.json" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
obj = json.loads(path.read_text(encoding="utf-8"))
short = obj.get("short_tft_student", {})
print(short.get("val_mase", "nan"))
print(short.get("metrics", {}).get("MASE", "nan"))
print(short.get("model_dir", ""))
PY
)

  VAL_MASE="${METRICS[0]}"
  TEST_MASE="${METRICS[1]}"
  MODEL_DIR="${METRICS[2]}"
  echo "$W,$VAL_MASE,$TEST_MASE,$RUN_DIR,$MODEL_DIR,ok" >> "$SUMMARY_CSV"
  echo "teacher_weight=$W done | val_mase=$VAL_MASE | test_mase=$TEST_MASE"
done

echo
echo "==== [2/3] Select best short model by test_mase ===="
readarray -t BEST < <(python - "$SUMMARY_CSV" <<'PY'
import csv, math, sys
from pathlib import Path

path = Path(sys.argv[1])
best = None
with path.open("r", encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        if row.get("status") != "ok":
            continue
        try:
            test_mase = float(row["test_mase"])
        except Exception:
            continue
        if not math.isfinite(test_mase):
            continue
        if best is None or test_mase < best[0]:
            best = (test_mase, row)

if best is None:
    raise SystemExit("No successful short runs in grid.")

row = best[1]
print(row["teacher_weight"])
print(row["run_dir"])
print(row["model_dir"])
print(row["test_mase"])
PY
)

BEST_W="${BEST[0]}"
BEST_RUN_DIR="${BEST[1]}"
BEST_MODEL_DIR="${BEST[2]}"
BEST_TEST_MASE="${BEST[3]}"

echo "Best teacher_weight=$BEST_W | short test_mase=$BEST_TEST_MASE"
echo "Best model dir: $BEST_MODEL_DIR"

echo
echo "==== [3/3] Iterative rollout with best short model ===="
BEST_ITER_DIR="$OUT_ROOT/best_iterative"
mkdir -p "$BEST_ITER_DIR"

PYTHONUNBUFFERED=1 python -u "$ITER_SCRIPT" \
  --gpu-id "$GPU_ID" \
  --input-dir "$INPUT_DIR" \
  --metadata-path "$META_PATH" \
  --output-root "$BEST_ITER_DIR" \
  --log-path "$BEST_ITER_DIR/iter_log.json" \
  --short-horizon-hours 6 \
  --long-horizons 12,24,48,72,120 \
  --context-hours 168 \
  --window-stride-hours 6 \
  --val-max-windows-per-station 64 \
  --test-max-windows-per-station 64 \
  --rollout-batch-size 32 \
  --rollout-past-covariates actual \
  --sample-windows-per-split 10 \
  --pretrained-tft-path "$BEST_MODEL_DIR" \
  --pretrained-tft-model-name TemporalFusionTransformer \
  --verbosity 3 \
  > "$BEST_ITER_DIR/run.log" 2>&1

echo
echo "All done."
echo "Short grid summary: $SUMMARY_CSV"
echo "Best iterative metrics: $BEST_ITER_DIR/iterative_metrics_summary.csv"
echo "Best iterative log: $BEST_ITER_DIR/iter_log.json"
