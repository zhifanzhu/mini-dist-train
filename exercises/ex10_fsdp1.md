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

This ex10 is extremely hard for me. I asked codex for hint, and essentially I need the following non-obvious operation:

```
self.flat_param = nn.Parameter(local_shard.clone())

full_flat = unshard().detach().requires_grad_(True)
owner._parameters[local_name] = view_into_full_flat

#  Do not use original_param.data = view; that disconnects the view from full_flat’s autograd graph. After backward, restore those ._parameters entries to None. This internal parameter swapping is the advanced PyTorch mechanism ex10 is teaching.
```

Okay I didn't know the existence of `_parameters` function.


I skipped ex10. I think the solution of ex10 departs the actual implementation.