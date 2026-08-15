#!/usr/bin/env bash
set -euo pipefail
pytest tests/gpu -q -m nccl
