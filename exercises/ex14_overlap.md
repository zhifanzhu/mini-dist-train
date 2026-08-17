# ex14 — Overlap, CUDA streams, prefetch

GPU extension. Launch async collectives, reason about work handles/stream dependencies, and prefetch the next parameter group without consuming an unready buffer. Correctness first, then timing.

## CUDA/PyTorch navigation hints

- Use one spawned process per GPU and call `torch.cuda.set_device(local_rank)`
  before initializing the NCCL process group.
- `dist.all_reduce(..., async_op=True)` returns a `Work` handle. Keep both the
  handle and communication buffer alive until completion; launching async and
  immediately synchronizing gives correctness but no overlap.
- Create a separate `torch.cuda.Stream` for communication. Use
  `communication_stream.wait_stream(default_stream)` before communication if
  the buffer was produced on the default stream.
- Record a `torch.cuda.Event` after communication completion and make the
  consumer stream call `wait_event()` before reading the result. A Python
  `Work.wait()` and a CUDA stream dependency answer different readiness
  questions; reason about both.
- Enqueue independent default-stream computation between launch and the final
  dependency. Avoid a device-wide `torch.cuda.synchronize()` until measurement
  or final verification, because it destroys the overlap you are trying to
  observe.

The probe fields are explicit acceptance invariants: correct numerics, an
asynchronous launch, a separate stream, and a wait before consumption.
