# ex02 — Ring all-reduce (optional branch)

Implement `ring_all_reduce_sum()` using only point-to-point communication. Blocking, nonblocking, or batched P2P APIs are all valid. Implement the standard chunked reduce-scatter and all-gather phases. This is for communication intuition; later framework code should still use production collectives via `torch.distributed`.

The tests check numerical results across several group shapes but intentionally do not intercept particular PyTorch P2P calls, because doing so would reject valid implementations using `send`/`recv`, `isend`/`irecv`, or batched operations. After the tests pass, ask Codex or another code-review agent to verify that the implementation really has the two ring phases and communicates chunks rather than full tensors. This algorithm-structure review is part of ex02.

## PyTorch navigation hints

- `dist.get_rank(group)` returns a rank local to the group. With the classic
  `src=`/`dst=` P2P arguments, peers are global ranks; translate a group-local
  neighbor with `dist.get_global_rank(group, group_rank)` for a non-contiguous
  subgroup.
- A simple deadlock-safe nonblocking pattern is to post both `isend` and
  `irecv`, wait for the receive before consuming its buffer, and wait for the
  send before reusing its source buffer. Other equivalent schedules are valid.
- `Tensor.chunk()` returns views into the original flat tensor. Updating a
  received/reduced chunk in place can therefore update the caller's tensor
  without a final concatenate.
