# ex10 — FSDP1-style lifecycle

Wrap the flat parameter with pre/post forward/backward behavior. The public training loop should look like normal PyTorch while hooks drive all-gather, reshard, and gradient reduce-scatter.

## PyTorch framework hints

This chapter intentionally uses parameter-registration internals. The wrapper
must expose exactly one optimizer-visible local flat `nn.Parameter`; the
original parameters must not also appear in `model.parameters()`. Resolve each
qualified parameter name to its owning submodule and temporarily install
original-shaped tensor views backed by the gathered full flat buffer. One
acceptable educational route is to manage the owner's `_parameters` entry
directly; ordinary `setattr` will reject assigning a plain tensor where a
registered `nn.Parameter` already exists.

A workable lifecycle is:

1. Before wrapped computation, all-gather the local flat parameter and create
   an autograd-enabled full flat tensor.
2. Install ex09's views into their original module slots and run the module.
3. Register a tensor autograd hook with `full_flat.register_hook(...)`. Its
   incoming full-flat gradient can be padded and reduce-scattered into
   `flat_param.grad` as a mean local shard.
4. When resharding, remove the temporary views and release the full buffer.

Do not create the optimizer until after wrapping, or it may retain references
to the original full parameters. Also ensure the wrapped module is not
accidentally registered in a way that makes those originals visible alongside
the flat parameter.
