# ex12 — FSDP2-style parameter groups

Group multiple `FSDPParam` objects for communication. Pack shards into a temporary collective buffer, all-gather once, then restore individual parameter views. Backward performs grouped reduce-scatter and returns rank-local shards of the mean gradient.

The tests check reconstructed values, gradient shards, padding, and state transitions without requiring a particular collective API. After they pass, ask Codex or another code-review agent to verify that parameters are temporarily packed for one grouped all-gather and one grouped reduce-scatter rather than communicated one by one. This grouped-communication review is part of ex12.

## Files to implement

- `mini_dist/fsdp2/param_group.py`
