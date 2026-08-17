#!/usr/bin/env bash
set -euo pipefail

pytest tests/exercises/test_ex17_tensor_parallel_linear.py -q
pytest tests/exercises/test_ex18_tensor_parallel_mlp.py -q
pytest tests/exercises/test_ex19_hybrid_parallel.py -q
