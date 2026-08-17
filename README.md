# mini-dist-train

A cumulative, test-driven distributed-training course that grows one small framework from raw process groups into DDP, ZeRO, FSDP1-style, and FSDP2-style training.

This repository is intentionally **not** a collection of independent toy scripts. Later exercises import and depend on earlier implementations, so bugs in sharding, padding, bucket metadata, ownership, or collective ordering can surface much later in the stack—closer to a real distributed framework.

## What you will be able to do at the end

After completing the core track, you should be able to:

- explain the SPMD/process-group model used by `torch.distributed`;
- reason precisely about `all_reduce`, `all_gather`, `reduce_scatter`, `broadcast`, and point-to-point send/recv;
- implement small ring and tree all-reduces from point-to-point operations and understand why production frameworks normally delegate collectives to communication backends;
- implement replicated data parallel training with gradient synchronization, autograd hooks, buckets, and asynchronous communication;
- derive and implement the memory/communication progression from DDP to ZeRO-1, ZeRO-2, and ZeRO-3;
- explain why ZeRO-1 shards optimizer state, ZeRO-2 additionally shards gradients, and ZeRO-3 additionally shards parameters;
- understand the shared algorithmic core behind DeepSpeed ZeRO-3 and PyTorch FSDP;
- implement an FSDP1-style flat-parameter representation, including flatten → pad → shard → all-gather → unpad → view and exact metadata tracking;
- implement an FSDP2-style per-parameter sharded representation with parameter groups and temporary communication packing rather than a permanent flat parameter;
- compare a DeepSpeed-style engine abstraction with PyTorch-style module/autograd integration;
- reason about parameter, gradient, and optimizer-state ownership at every point in forward/backward;
- debug collective-order mismatches, rank exits, skipped collectives, stragglers, and pre-collective exceptions;
- understand where ProcessGroupNCCL/NCCL sits below PyTorch and DeepSpeed, and use NCCL logging/async behavior in GPU labs;
- run the same model under baseline, DDP, ZeRO, FSDP1-style, and FSDP2-style implementations and explain why the numerical updates agree while memory ownership and communication differ.

The extended tensor-parallel track additionally reconstructs column-wise and
row-wise linear layers, a feature-sharded MLP, and a two-dimensional TP+FSDP
process mesh using explicit autograd-aware collectives.

The intended outcome is not "I know the FSDP API." It is: **I can reconstruct why sharded-training frameworks have the abstractions and failure modes that they do.**

## Hardware requirements

### Tier A — CPU-only: enough for the main correctness curriculum

Recommended minimum:

- Linux preferred (macOS works for many Gloo exercises; Windows is not the primary target);
- Python 3.10+;
- PyTorch 2.4+;
- 4 CPU cores recommended;
- 8 GB RAM recommended.

With CPU + Gloo you can complete the majority of the conceptual/core exercises:

- process groups and SPMD;
- send/recv and collective semantics;
- ring and tree all-reduce;
- MiniDDP correctness;
- bucket packing and metadata;
- ZeRO-1/2/3 correctness;
- FSDP1-style flattening/sharding;
- FSDP2-style per-parameter sharding;
- tensor-parallel linear/MLP correctness and four-process TP+FSDP composition;
- most fault-injection exercises;
- end-to-end numerical-equivalence tests.

You do **not** need an H100 or a multi-GPU server to learn the core algorithms.

### Tier B — 2 CUDA GPUs: recommended for the full systems track

Two NVIDIA GPUs on one machine are sufficient for:

- real NCCL-backed collectives;
- CUDA/NCCL asynchronous execution experiments;
- communication/computation overlap;
- CUDA stream exercises;
- NCCL failure/debug logging;
- basic direct-NCCL experiments.

The GPUs do not need to be datacenter GPUs. Two reasonably modern CUDA-capable GPUs are enough for the educational workloads.

### Tier C — 4 CUDA GPUs: strongly recommended, not required

Four GPUs make these topics much more visible:

- nontrivial shard ownership;
- uneven padding cases across more ranks;
- gradient/parameter bucket behavior;
- overlap and prefetch experiments;
- collective ordering/debugging;
- topology-sensitive NCCL observations.

### Tier D — multi-node: optional extension only

Two or more machines are useful if you specifically want to study:

- inter-node NCCL behavior;
- network transport/topology;
- multi-node stragglers and failures;
- InfiniBand/RDMA environments.

Multi-node hardware is **not required** for the main course.

### Optional direct-NCCL C++ lab

For the direct NCCL lab (outside PyTorch), you additionally need:

- NVIDIA driver;
- CUDA toolkit/compiler;
- NCCL development headers and library.

PyTorch having NCCL runtime support does not necessarily mean the system NCCL development headers are installed.

## Course shape

The dependency spine is:

```text
process groups
    ↓
collectives ─────────────→ optional ring/tree all-reduce labs
    ↓
MiniDDP
    ↓
ZeRO-1
    ↓
ZeRO-2
    ↓
ZeRO-3
    ↓
framework realizations
    ├── DeepSpeed-style Engine
    ├── FSDP1-style flat parameter
    └── FSDP2-style per-parameter state + param groups
    ↓
overlap / streams / prefetch
    ↓
NCCL + failure/debugging
    ↓
final equivalence + memory/communication comparison
```

The ring and tree labs are intentionally side branches: they compare two ways to build a collective from point-to-point communication, but the main framework uses `torch.distributed` collectives like real PyTorch/DeepSpeed implementations do.

The optional tensor-parallel branch starts from collectives and rejoins the
FSDP path in a two-dimensional mesh:

```text
collectives → TP linear styles → TP MLP ─────┐
                                             ├→ 2D TP + FSDP
ZeRO-3 → FSDP2 → fully_shard ────────────────┘
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Check that the course harness itself works:

```bash
pytest tests/smoke -q
```

The smoke suite should pass before you implement anything.

## How to work through an exercise

Read `course.yml` or `exercises/README.md`, then run exactly one chapter, for example:

```bash
pytest tests/exercises/test_ex03_ddp.py -q
```

The first unimplemented TODO should fail. Implement only the requested part, rerun that chapter, then run all previous chapters as a regression suite.

For example:

```bash
pytest tests/exercises/test_ex00_process_groups.py \
       tests/exercises/test_ex01_collectives.py \
       tests/exercises/test_ex03_ddp.py -q
```

Later exercises are cumulative and deliberately reuse earlier code.

## Codex workflow

Open this repository in Codex from the repository root. `AGENTS.md` tells Codex to act as a test/debugging partner rather than solving exercises automatically.

Useful prompts:

```text
Run the current chapter tests. Do not edit my implementation.
Tell me which invariant failed and give one minimal hint.
```

```text
This distributed test hangs. Re-run it with the repository's timeout/debug tooling,
inspect the rank behavior, and explain where collective progress diverged.
Do not fix the code for me.
```

```text
Run all completed chapters as a regression suite. Do not modify code.
```

## Repository layout

```text
mini_dist/
    distributed.py
    collectives.py
    buckets.py
    ddp.py
    zero/
    fsdp1/
    fsdp2/
    tensor_parallel/
    engine.py

exercises/
    README.md
    ex00_...md ... ex19_...md

tests/
    smoke/
    exercises/
    gpu/
    milestones/

fault_injection/
    skipped_collective.py
    mismatched_collective.py
    rank_exit.py
    straggler.py
    pre_collective_exception.py

nccl/
    README.md
    direct_allreduce.cu

scripts/
    progress.py
    run_cpu_core.sh
    run_gpu_nccl.sh
```

## No solutions are included

The repository provides interfaces, TODOs, tests, invariants, debugging utilities, and milestone programs. It intentionally does not include reference implementations of the exercises.
