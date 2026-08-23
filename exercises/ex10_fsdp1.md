# ex10 — FSDP1-style lifecycle

Wrap the flat parameter with pre/post forward/backward behavior. The public training loop should look like normal PyTorch while hooks drive all-gather, reshard, and gradient reduce-scatter.

This exercise keeps the gathered flat parameter through backward, matching `SHARD_GRAD_OP`. Production `FULL_SHARD` frees it after forward and all-gathers it again before backward; that storage-lifetime machinery is out of scope.

## Files to implement

- `mini_dist/fsdp1/fsdp.py`

## Hints

- `Tensor.register_hook()` on the result of `FlatParameterHandle.unshard()`

## My notes

SHARD_GRAD_OP: https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.ShardingStrategy

Q: In https://docs.pytorch.org/tutorials/intermediate/FSDP1_tutorial.html, it says 
"Discard parameter shards it has just collected". Is this happening in this tutorial? If not, what stops us from letting it happen? Otherwise, this is not a faithful implmenetation of FSDP's doc.

Okay I looked at the text_ex10_fsdp1.py. The instructions are not really unclear.