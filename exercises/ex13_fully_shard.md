# ex13 — `fully_shard()`-style API

Implement bottom-up FSDP ownership. Child modules already sharded must not be re-owned by a parent group. Register hooks so the final user-facing API composes with ordinary modules/optimizers.

## Files to implement

- `mini_dist/fsdp2/fully_shard.py`
