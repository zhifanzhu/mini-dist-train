# ex10 — FSDP1-style lifecycle

Wrap the flat parameter with pre/post forward/backward behavior. The public training loop should look like normal PyTorch while hooks drive all-gather, reshard, and gradient reduce-scatter.

## Files to implement

- `mini_dist/fsdp1/fsdp.py`


## My notes

Q: In https://docs.pytorch.org/tutorials/intermediate/FSDP1_tutorial.html, it says 
"Discard parameter shards it has just collected". Is this happening in this tutorial? If not, what stops us from letting it happen? Otherwise, this is not a faithful implmenetation of FSDP's doc.