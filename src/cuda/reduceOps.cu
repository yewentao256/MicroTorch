#include "ops.hpp"

namespace microtorch {

__global__ void sum(size_t n, float *a, float *out) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = index; i < n; i += stride) {
    // Note: multi-thread write here, we should use atomicAdd
    atomicAdd(&out[0], a[i]);
  }
}

template <>
void sum_impl<Cuda>(Tensor &a, Tensor &out) {
  float *a_ptr = a.data_ptr();
  float *out_ptr = out.data_ptr();

  size_t blockSize = 256;
  size_t numBlocks = (a.numel() + blockSize - 1) / blockSize;  // Ceilling
  sum<<<numBlocks, blockSize>>>(a.numel(), a_ptr, out_ptr);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
  }
}

}  // namespace microtorch
