# Direct NCCL lab (optional)

This directory deliberately steps below `torch.distributed` once.

Goals:

1. initialize an NCCL communicator for one process per GPU;
2. call `ncclAllReduce` on a CUDA stream;
3. reason about host enqueue vs CUDA-stream completion;
4. compare the result/contract with `torch.distributed.all_reduce` using ProcessGroupNCCL;
5. optionally extend the program to `ncclAllGather`, `ncclReduceScatter`, and `ncclSend`/`ncclRecv`.

Requirements: CUDA toolkit, NCCL development headers/library, and at least 2 NVIDIA GPUs.

`direct_allreduce.cu` is a compileable scaffold with TODO markers, not a solution.
