# ex06 — ZeRO-2

Add sharded reduced gradients. Use reduce-scatter semantics so optimizer-state ownership and gradient ownership align. Parameters are still replicated after the optimizer step.
