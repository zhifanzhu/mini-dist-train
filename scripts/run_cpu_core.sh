#!/usr/bin/env bash
set -euo pipefail
pytest tests/smoke -q
pytest tests/exercises/test_ex00_process_groups.py -q
pytest tests/exercises/test_ex01_collectives.py -q
pytest tests/exercises/test_ex03_ddp.py -q
pytest tests/exercises/test_ex04_buckets.py -q
pytest tests/exercises/test_ex05_zero1.py -q
pytest tests/exercises/test_ex06_zero2.py -q
pytest tests/exercises/test_ex07_zero3.py -q
pytest tests/exercises/test_ex08_engine.py -q
pytest tests/exercises/test_ex09_fsdp1_flat_param.py -q
pytest tests/exercises/test_ex10_fsdp1.py -q
pytest tests/exercises/test_ex11_fsdp2_param.py -q
pytest tests/exercises/test_ex12_fsdp2_param_group.py -q
pytest tests/exercises/test_ex13_fully_shard.py -q
pytest tests/milestones/test_equivalence.py -q
