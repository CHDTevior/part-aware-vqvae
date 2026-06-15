#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PARTITION_FILE="${PARTITION_FILE:-./partition_analysis/skeleton_partition.json}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -u train_vq.py \
  --dataname t2m \
  --seed 123 \
  --exp-name vq_overlap_top3_20260529 \
  --nb-code 128 \
  --partition-file "${PARTITION_FILE}" \
  "$@"
