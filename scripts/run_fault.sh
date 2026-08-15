#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "usage: $0 <fault_injection/script.py>" >&2
  exit 2
fi
WORLD_SIZE="${WORLD_SIZE:-2}"
torchrun --standalone --nproc-per-node="$WORLD_SIZE" "$1"
