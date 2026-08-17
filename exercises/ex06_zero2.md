# ex06 — ZeRO-2

Add sharded reduced gradients. Use reduce-scatter semantics so optimizer-state ownership and gradient ownership align. Parameters are still replicated after the optimizer step.

## My notes

Q: so unlike ex05, here we do not assume a fully-reduced gradient? so it is unnecessary to use this together with DDP?
From this exercise, it seems that:
- ZeRO-1 + DDP is a good combination
- ZeRO-2 replaces DDP

## Lifecycle and collective hints

Unlike ex05, the intended `Zero2Optimizer` path receives rank-local full
gradients and owns their cross-rank synchronization; an external DDP
all-reduce is unnecessary. Pack local gradients in exactly the same parameter
order as the master parameter, pad to equal partitions, and reduce-scatter
corresponding chunks. Divide the SUM result by the process-group size to obtain
this rank's mean-gradient shard.

Taking only `start:end` from the unreduced local gradient is not enough: it
would update each parameter region from only its owner's minibatch. The
collective must combine that same region from every rank.

Assign the reduced shard to the local master parameter, update locally, then
reuse ex05's parameter all-gather so model parameters remain replicated. Keep
`local_grad_shard` as the observable owned gradient and reset it in
`zero_grad()`.
