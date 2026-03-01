#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/link_large_data.sh /abs/path/to/HumanML3D
# Or set HUMANML3D_DIR env var.

TARGET="${1:-${HUMANML3D_DIR:-}}"
if [ -z "${TARGET}" ]; then
  echo "Usage: $0 /abs/path/to/HumanML3D"
  echo "Or set HUMANML3D_DIR env var."
  exit 1
fi

if [ ! -d "${TARGET}" ]; then
  echo "Target dataset directory not found: ${TARGET}"
  exit 1
fi

mkdir -p dataset
ln -sfn "${TARGET}" dataset/HumanML3D

echo "Linked dataset/HumanML3D -> ${TARGET}"
