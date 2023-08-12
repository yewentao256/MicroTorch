#include "ops.hpp"

namespace tinytorch {

__global__ void fill(size_t n, float *self, float value) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = index; i < n; i += stride) {
    self[i] = value;
  }
}

template <>
void fill_impl<Cuda>(Tensor &self, const data_t value) {
  float *self_ptr = self.data_ptr();

  size_t blockSize = 256;
  size_t numBlocks = (self.numel() + blockSize - 1) / blockSize;  // Ceilling
  fill<<<numBlocks, blockSize>>>(self.numel(), self_ptr, value);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
  }
}

}  // namespace tinytorch
