#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python train.py \
  --dataset nsl \
  --initial-ratio 0.10 \
  --initial-epochs 10 \
  --online-epochs 2 \
  --window-size 2000 \
  --memory-size 4096 \
  --replay-batch-size 256 \
  --tau 0.85 \
  --consistency-tau 0.90
