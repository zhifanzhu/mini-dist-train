# ex06 — ZeRO-2

Add sharded reduced gradients. Use reduce-scatter semantics so optimizer-state ownership and gradient ownership align. Parameters are still replicated after the optimizer step.

## Files to implement

- `mini_dist/zero/stage2.py`

## My notes

Q: so unlike ex05, here we do not assume a fully-reduced gradient? so it is unnecessary to use this together with DDP?
From this exercise, it seems that:
- ZeRO-1 + DDP is a good combination
- ZeRO-2 replaces DDP
  - ZeRO-2 already covers the job of DDP. ZeRO-2 receives the full gradient of a local batch, it reduce_scatter this gradient. The reduce does the DDP's sync_gradient's job. The scatter then ensures each rank obtain the shard of the gradient they own.
  - In my stage2.py I did not implement bucketing as did in miniDDP. My code actually materialise the full gradient during `reduce_scatter_gradient()`. In practice ZeRO-2 should also register to a backward_hook and do bucketing I guess?


### bucketing

For both DDP and ZeRO-2, bucketing matters. Consider the cycle

fwd — bwd — optim.step — fwd — bwd ...

- DDP holds full gradient in both bwd and optim.step.
- In worst-case bucketing, ZeRO-2 hold full gradient also in bwd -- so same peak memory as in DDP -- but hold only local shard during optim.step.
- In typical scenarios when bucketing works, ZeRO-2 also reduce memory use in bwd.

