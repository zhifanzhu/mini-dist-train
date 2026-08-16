# Direct NCCL lab (optional)

This directory deliberately steps below `torch.distributed` once.

Goals:

1. initialize an NCCL communicator for one process per GPU;
2. call `ncclAllReduce` on a CUDA stream;
3. reason about host enqueue vs CUDA-stream completion;
4. compare the result/contract with `torch.distributed.all_reduce` using ProcessGroupNCCL;
5. optionally extend the program to `ncclAllGather`, `ncclReduceScatter`, and `ncclSend`/`ncclRecv`.

Requirements: CUDA toolkit, NCCL development headers/library, and at least 2 NVIDIA GPUs.

`direct_allreduce.cu` contains a complete single-process, multi-GPU realization using `ncclCommInitAll`. This avoids adding MPI solely for communicator-ID exchange while exercising the same communicator, CUDA-stream enqueue, completion, and result contracts. Extending it to one process per GPU replaces `ncclCommInitAll` with an external `ncclUniqueId` exchange and `ncclCommInitRank`.

One example build command is:

```bash
nvcc -O2 direct_allreduce.cu -lnccl -o direct_allreduce
./direct_allreduce 2
```
