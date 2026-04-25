#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/ssd/lian/给claudecode/v10_baseline_conpoff"
WORKDIR="$ROOT/用其他的轮计算"
SCRIPT="$WORKDIR/run_fast_checkpoint_eval_cpu_priority.py"
CKPT_DIR="$ROOT/run/20260422_201016_COnP/checkpoints"
LOG="$WORKDIR/batch_fast_80_500.log"

mkdir -p "$WORKDIR"

mapfile -t checkpoints < <(
  python3 - <<'PY'
from pathlib import Path
import re
ckpt_dir=Path('/mnt/ssd/lian/给claudecode/v10_baseline_conpoff/run/20260422_201016_COnP/checkpoints')
pat=re.compile(r'best_model_epoch(\d+)_COnP')
rows=[]
for p in sorted(ckpt_dir.glob('best_model_epoch*_COnP*.pt')):
    m=pat.search(p.name)
    if m:
        ep=int(m.group(1))
        if 80 <= ep <= 500:
            rows.append(p.name)
for x in rows:
    print(x)
PY
)

{
  echo "=== batch start $(date '+%F %T') ==="
  echo "root=$ROOT"
  echo "script=$SCRIPT"
  echo "checkpoint_dir=$CKPT_DIR"
  echo "count=${#checkpoints[@]}"
  echo
} | tee -a "$LOG"

total="${#checkpoints[@]}"
idx=0
for ckpt_name in "${checkpoints[@]}"; do
  idx=$((idx + 1))
  ckpt="$CKPT_DIR/$ckpt_name"
  echo "[${idx}/${total}] START $ckpt_name $(date '+%F %T')" | tee -a "$LOG"
  python3 "$SCRIPT" "$ckpt" --fixed-frame 0.40 2>&1 | tee -a "$LOG"
  echo "[${idx}/${total}] END   $ckpt_name $(date '+%F %T')" | tee -a "$LOG"
  echo | tee -a "$LOG"
done

echo "=== batch end $(date '+%F %T') ===" | tee -a "$LOG"
