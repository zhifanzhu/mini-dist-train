# Project contract / design lock

This file records the design decisions for this educational repository so a coding agent does not silently turn it into a different course.

1. **One cumulative framework.** Exercises are mostly dependent. Later chapters import earlier implementations.
2. **Cross-framework, not PyTorch-only.** The conceptual spine is DDP → ZeRO-1 → ZeRO-2 → ZeRO-3, followed by DeepSpeed-style, FSDP1-style, and FSDP2-style realizations.
3. **Implement both FSDP cores.** FSDP1-style teaches permanent flattening/views/offsets; FSDP2-style teaches per-parameter identity/state and temporary grouped communication packing.
4. **Padding is mandatory.** Non-divisible tensor/buffer lengths are not optional edge cases. Track exact original shape/numel/offset/padded length and reconstruct exactly.
5. **Stay close to production abstraction boundaries.** Framework layers call `torch.distributed` collectives; the ring implementation is an educational side lab, not the communication substrate for MiniFSDP/ZeRO.
6. **NCCL is part of the systems track.** CPU/Gloo is used for correctness; GPU/NCCL labs cover async work, CUDA streams, overlap, debug logs, collective ordering, and failure propagation. Direct NCCL C++ is an optional small lab; implementing NCCL itself is out of scope.
7. **Failure injection is required.** Include rank exit, skipped collective, mismatched collective, straggler, and exception/OOM-before-collective scenarios.
8. **Tests assert invariants, not only final output.** Ownership, padding metadata, state transitions, collective semantics, and cross-rank equality should be observable/testable.
9. **Normal final APIs should look boring.** FSDP-style usage should converge toward an ordinary PyTorch training loop; DeepSpeed-style usage should expose an engine abstraction.
10. **No bundled solutions.** Codex/test tooling should run and diagnose, but not solve TODOs unless the learner explicitly asks.
11. **Tests are the acceptance boundary.** Every required correctness property belongs in the public test suite. Chapter tests use a compact representative matrix, with subgroup behavior tested at the primitive communication layer, and a passing implementation is accepted without agent-invented stress tests.
12. **Keep the weekend-project scope.** Do not turn rare, expensive, backend-specific, or substantially complex corners into mandatory requirements. Put them in optional exercises or document them as limitations instead.
13. **Do not test implementation spelling.** Tests assert public behavior and exposed educational invariants, not a particular equivalent API choice such as blocking versus nonblocking P2P. If one central algorithmic property cannot be tested without rejecting valid designs, mark that property narrowly as agent-reviewed in both the exercise brief and relevant code; otherwise drop the check.
14. **Tensor parallelism is an extended branch.** ex17–19 reconstruct the explicit layout algebra beneath PyTorch DTensor and JAX sharding APIs: column/row linear styles, a feature-sharded MLP, and a 2D TP+FSDP mesh. Keep the required track CPU/Gloo-testable; sequence, attention, pipeline, expert, and loss parallelism remain follow-up extensions.
