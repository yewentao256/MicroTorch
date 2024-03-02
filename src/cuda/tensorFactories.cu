/**
 * Copyright (c) 2022-2023 yewentao
 * Licensed under the MIT License.
 */
#include <curand_kernel.h>

#include "cuda.hpp"
#include "functors.hpp"
#include "loops.cuh"
#include "ops.hpp"

namespace microtorch {

__global__ void fill_kernel(int64_t n, float *self, float value) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    self[i] = value;
  }
}

template <>
void fill_impl<Cuda>(Tensor &self, const data_t value) {
  int64_t blocks_per_grid = get_blocks_per_grid(self.numel());
  fill_kernel<<<blocks_per_grid, ThreadsPerBlock>>>(self.numel(),
                                                    self.data_ptr(), value);
  CUDA_ERROR_CHECK();
}

template <>
void clone_impl<Cuda>(TensorIterator &iter) {
  gpu_kernel(iter, binaryFunctor::Clone());
}

__global__ void rand_kernel(float *data, int numel) {
  int64_t stride = blockDim.x * gridDim.x;

  // For each block, initialize it's own state
  __shared__ curandState state;
  if (threadIdx.x == 0) {
    curand_init(1234, blockIdx.x, 0, &state);
  }
  __syncthreads();

  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numel; i += stride) {
    data[i] = curand_uniform(&state);
  }
}

template <>
void rand_impl<Cuda>(Tensor &self) {
  int64_t blocks_per_grid = get_blocks_per_grid(self.numel());
  rand_kernel<<<blocks_per_grid, ThreadsPerBlock>>>(self.data_ptr(),
                                                    self.numel());
  CUDA_ERROR_CHECK();
}

}  // namespace microtorch
