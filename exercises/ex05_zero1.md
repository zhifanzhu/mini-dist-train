# ex05 — ZeRO-1

Shard optimizer ownership while parameters and gradients remain replicated. Each rank updates only its owned parameter slice; updated slices must be reconstructed so every rank sees the same full parameter after `step()`.

## Files to implement

- `mini_dist/zero/common.py`
- `mini_dist/zero/stage1.py`
