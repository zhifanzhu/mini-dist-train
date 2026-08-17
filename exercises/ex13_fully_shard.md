# ex13 — `fully_shard()`-style API

Implement bottom-up FSDP ownership. Child modules already sharded must not be re-owned by a parent group. Register hooks so the final user-facing API composes with ordinary modules/optimizers.

## Ownership and hook hints

This is hierarchy ownership, not forward-order discovery. The caller applies
`fully_shard()` bottom-up (children before parent), as shown in the test. When
sharding a parent, inspect already-sharded descendants, collect their owned
parameter identities, and exclude those objects from the parent's group.
Calling `fully_shard()` twice on the same module should be idempotent.

The two useful PyTorch mechanisms are:

- `module.register_forward_pre_hook(...)` to clear readiness and unshard this
  module's owned group immediately before its computation;
- `parameter.register_post_accumulate_grad_hook(...)` to mark an owned leaf
  ready only after `.grad` exists. Once every owned parameter is ready, invoke
  ex12's grouped reduce-scatter, install local gradient shards, and restore
  sharded parameter storage/state.

Hooks registered in a loop must capture the current `FSDPParam` with a factory
or default argument. Preserve original parameter identities throughout and
construct the optimizer only after `fully_shard()`, when parameter storage is
local. A module with no newly owned parameters should still compose safely.
