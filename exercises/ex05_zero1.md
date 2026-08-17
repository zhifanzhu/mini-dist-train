# ex05 — ZeRO-1

Shard optimizer ownership while parameters and gradients remain replicated. Each rank updates only its owned parameter slice; updated slices must be reconstructed so every rank sees the same full parameter after `step()`.


FSDP1 tutorial is much more well-written than FSDP2.
https://docs.pytorch.org/tutorials/intermediate/FSDP1_tutorial.html

For ZeRO, reference is:
https://deepspeed.readthedocs.io/en/latest/zero3.html

Q: how will Zero1Optimizer be called? Will I learn it?
Ans: check test_ex05_zero1.py to see how this optimizer is going to be used.
- Q: is that actually how it is used? Ans: It seems so...