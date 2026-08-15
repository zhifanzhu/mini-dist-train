# Testing philosophy

The tests are intentionally cumulative and invariant-oriented.

They are also the acceptance specification for the exercises. Required
behavior belongs in public tests; a learner whose chapter and regression tests
pass should not need to satisfy additional agent-invented stress cases.

Tests should not monkeypatch or count one specific low-level API when equivalent
blocking, nonblocking, batched, or backend-specific implementations satisfy the
same exercise contract. A central algorithmic property that cannot be observed
without such coupling may be marked narrowly as agent-reviewed in the exercise
brief and source scaffold. Unimportant unverifiable details are simply omitted.

## Two categories

### Smoke tests

```bash
pytest tests/smoke -q
```

These validate the repository/test harness and should pass in the untouched starter repository.

### Exercise tests

These are expected to fail at TODOs until implemented:

```bash
pytest tests/exercises/test_ex00_process_groups.py -q
```

After a chapter passes, keep it in your regression set when working on later chapters.

## What tests should establish

Tests should prefer meaningful distributed invariants over black-box success:

- exact rank/world-size state;
- SUM vs MEAN reduction semantics;
- equal shard size after padding;
- exact original shape/numel after reconstruction;
- replicated parameters agree after DDP/ZeRO-1/2 updates;
- only the intended gradient/optimizer shard is retained;
- FSDP parameters transition through the correct states;
- grouped communication does not destroy per-parameter identity;
- child FSDP groups are not re-owned by a parent;
- final numerical updates match a non-sharded reference.

## Hangs

A hanging distributed job is often a *semantic failure*, not a pytest bug. For failure labs, use short timeouts and reason about the ordered collective stream per rank.

## Required representative cases

The suite keeps a compact set of cases that exercise the declared contracts:

- `numel < world_size`;
- `numel % world_size != 0`;
- parameters of very different sizes in one communication group;
- world size 1 where practical;
- a non-default process group at the primitive communication layer;
- different minibatches per rank while initial parameters remain identical.

Rare backend-specific cases and features that require substantial extra
machinery, such as production-grade unused-parameter discovery, belong in an
optional extension or documented limitation unless a chapter explicitly adds
them to its tests.
