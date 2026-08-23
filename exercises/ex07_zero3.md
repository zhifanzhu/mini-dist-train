# ex07 — ZeRO-3

Add parameter sharding. Implement flatten/pad/shard metadata for the generic ZeRO-3 core and all-gather parameters around computation. After backward, parameters and mean-reduced gradients must be sharded again. This is the common algorithmic substrate for later FSDP-style realizations.

## Files to implement

- `mini_dist/zero/stage3.py`

## Hints

- Is the shard updated by the optimizer the one gathered by the next forward?

## My Notes 

Q: what about backward() pass? I guess it is omitted for educational purpose?
Will my operation require additional backward() impl?
Ans: Check ex08.