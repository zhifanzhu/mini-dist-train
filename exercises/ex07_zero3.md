# ex07 — ZeRO-3

Add parameter sharding. Implement flatten/pad/shard metadata for the generic ZeRO-3 core and all-gather parameters around computation. After backward, parameters and mean-reduced gradients must be sharded again. This is the common algorithmic substrate for later FSDP-style realizations.

## Files to implement

- `mini_dist/zero/stage3.py`

## Hints

- Is the shard updated by the optimizer the one gathered by the next forward?

## My Notes 

Q: what about backward() pass? I guess it is omitted for educational purpose?
Will my operation require additional backward() impl?
Ans: this is updated. See above hints.

In addition, I feel this is a much simplified exercise than actual ZeRO-3. In the actual ZeRO-3, submodules will be fit to memory in several phases during forward, but here the testcase only test for a single monolithic module.
- https://deepspeed.readthedocs.io/en/latest/zero3.html

I think for this exercise, ask Codex for hints are suggested.

Q: It seems that after forward(), the ZeRO3 still retain the full parameter? Is this the same came in production ZeRO-3? Or where do they drop the full param?