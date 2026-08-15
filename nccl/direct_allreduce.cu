#include <cuda_runtime.h>
#include <nccl.h>
#include <cstdio>

#define CUDA_CHECK(cmd) do { cudaError_t e = cmd; if (e != cudaSuccess) { \
  std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e)); return 1; }} while (0)
#define NCCL_CHECK(cmd) do { ncclResult_t r = cmd; if (r != ncclSuccess) { \
  std::fprintf(stderr, "NCCL error: %s\n", ncclGetErrorString(r)); return 1; }} while (0)

int main(int argc, char** argv) {
  // TODO ex15/direct-NCCL:
  // - choose a local rank/device (or extend to MPI/torchrun env vars)
  // - exchange a shared ncclUniqueId between processes
  // - ncclCommInitRank
  // - allocate/send a small CUDA buffer
  // - ncclAllReduce(..., stream)
  // - synchronize the stream and verify the result
  // - destroy communicator/free resources
  std::puts("TODO: direct NCCL exercise; read nccl/README.md");
  return 2;
}
