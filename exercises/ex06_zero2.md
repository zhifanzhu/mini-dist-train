# ex06 — ZeRO-2

Add sharded reduced gradients. Use reduce-scatter semantics so optimizer-state ownership and gradient ownership align. Parameters are still replicated after the optimizer step.

## Files to implement

- `mini_dist/zero/stage2.py`

## My notes

Q: so unlike ex05, here we do not assume a fully-reduced gradient? so it is unnecessary to use this together with DDP?
From this exercise, it seems that:
- ZeRO-1 + DDP is a good combination
- ZeRO-2 replaces DDP