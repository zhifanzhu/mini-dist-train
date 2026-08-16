#include <cuda_runtime.h>
#include <nccl.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(cmd) do { cudaError_t e = cmd; if (e != cudaSuccess) { \
  std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e)); return 1; }} while (0)
#define NCCL_CHECK(cmd) do { ncclResult_t r = cmd; if (r != ncclSuccess) { \
  std::fprintf(stderr, "NCCL error: %s\n", ncclGetErrorString(r)); return 1; }} while (0)

int main(int argc, char** argv) {
  int available = 0;
  CUDA_CHECK(cudaGetDeviceCount(&available));
  int nranks = argc > 1 ? std::atoi(argv[1]) : available;
  if (nranks < 2 || nranks > available) {
    std::fprintf(stderr, "usage: %s [number-of-local-gpus >= 2]; found %d GPUs\n", argv[0], available);
    return 2;
  }

  constexpr size_t count = 16;
  std::vector<int> devices(nranks);
  std::vector<ncclComm_t> communicators(nranks);
  std::vector<cudaStream_t> streams(nranks);
  std::vector<float*> buffers(nranks);
  for (int rank = 0; rank < nranks; ++rank) devices[rank] = rank;

  // ncclCommInitAll is the single-process equivalent of exchanging a
  // ncclUniqueId and calling ncclCommInitRank in one-process-per-GPU jobs.
  NCCL_CHECK(ncclCommInitAll(communicators.data(), nranks, devices.data()));
  for (int rank = 0; rank < nranks; ++rank) {
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(cudaStreamCreate(&streams[rank]));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffers[rank]), count * sizeof(float)));
    std::vector<float> input(count, static_cast<float>(rank + 1));
    CUDA_CHECK(cudaMemcpyAsync(buffers[rank], input.data(), count * sizeof(float),
                               cudaMemcpyHostToDevice, streams[rank]));
  }

  NCCL_CHECK(ncclGroupStart());
  for (int rank = 0; rank < nranks; ++rank) {
    NCCL_CHECK(ncclAllReduce(buffers[rank], buffers[rank], count, ncclFloat,
                             ncclSum, communicators[rank], streams[rank]));
  }
  NCCL_CHECK(ncclGroupEnd());

  const float expected = static_cast<float>(nranks * (nranks + 1) / 2);
  bool ok = true;
  for (int rank = 0; rank < nranks; ++rank) {
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(cudaStreamSynchronize(streams[rank]));
    std::vector<float> output(count);
    CUDA_CHECK(cudaMemcpy(output.data(), buffers[rank], count * sizeof(float), cudaMemcpyDeviceToHost));
    for (float value : output) ok = ok && std::fabs(value - expected) < 1e-5f;
  }

  for (int rank = 0; rank < nranks; ++rank) {
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(cudaFree(buffers[rank]));
    CUDA_CHECK(cudaStreamDestroy(streams[rank]));
    NCCL_CHECK(ncclCommDestroy(communicators[rank]));
  }
  std::printf("direct NCCL all-reduce: %s (%d ranks, expected %.1f)\n",
              ok ? "PASS" : "FAIL", nranks, expected);
  return ok ? 0 : 1;
}
