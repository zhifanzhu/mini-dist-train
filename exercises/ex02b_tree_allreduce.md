# ex02b — Tree all-reduce (optional branch)

Implement `tree_all_reduce_sum()` using only point-to-point communication. First combine partial sums through a balanced tree, then disseminate the complete sum through a tree. Blocking, nonblocking, and batched P2P APIs are all valid. Support arbitrary process-group sizes, including sizes that are not powers of two.

The tests check numerical results for a five-rank default group, a single rank, and a non-contiguous subgroup. They intentionally do not intercept particular PyTorch calls or prescribe a binary-heap, binomial, or other equivalent balanced-tree layout.

After the tests pass, ask Codex or another code-review agent to verify only the structural objective: both phases use point-to-point communication over a balanced tree rather than a linear chain, per-rank gather, or hidden collective. This narrow algorithm-structure review is part of ex02b.

## PyTorch navigation hints

- One useful way to reason about a balanced tree is a stride that doubles
  during reduction and halves during dissemination. Guard partners that fall
  outside `world_size`; this naturally handles non-powers of two.
- As in ex02, tree arithmetic is easiest in group-local ranks, while classic
  P2P `src=`/`dst=` arguments identify global ranks. Translate peers for a
  non-contiguous subgroup.
- A sender becomes inactive for the remaining reduction levels, but it must
  participate again at the matching dissemination level. Write each rank's
  ordered send/receive sequence before coding.
