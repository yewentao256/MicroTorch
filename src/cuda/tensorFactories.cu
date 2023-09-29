#include <curand_kernel.h>

#include "cuda.hpp"
#include "ops.hpp"

namespace microtorch {

__global__ void fill_kernel(int64_t n, float *self, float value) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;
  for (int64_t i = idx; i < n; i += stride) {
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

__global__ void clone_kernel(int64_t n, const float *a, float *out) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;
  for (int64_t i = idx; i < n; i += stride) {
    out[i] = a[i];
  }
}

template <>
void clone_impl<Cuda>(const Tensor &a, Tensor &out) {
  float *a_ptr = a.data_ptr();
  float *out_ptr = out.data_ptr();

  int64_t blocks_per_grid = get_blocks_per_grid(a.numel());
  clone_kernel<<<blocks_per_grid, ThreadsPerBlock>>>(a.numel(), a_ptr, out_ptr);
  CUDA_ERROR_CHECK();
}

__global__ void clone_backward_kernel(int64_t n, const float *grad_output_ptr,
                                      float *grad_input_ptr) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;
  for (int64_t i = idx; i < n; i += stride) {
    grad_input_ptr[i] = grad_output_ptr[i];
  }
}

template <>
void clone_backward_impl<Cuda>(const Tensor &grad_output, Tensor &grad_input) {
  float *grad_output_ptr = grad_output.data_ptr();
  float *grad_input_ptr = grad_input.data_ptr();

  int64_t blocks_per_grid = get_blocks_per_grid(grad_output.numel());
  clone_backward_kernel<<<blocks_per_grid, ThreadsPerBlock>>>(
      grad_output.numel(), grad_output_ptr, grad_input_ptr);
  CUDA_ERROR_CHECK();
}

__global__ void rand_kernel(float *data, int numel) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int64_t stride = blockDim.x * gridDim.x;

  // For each block, initialize it's own state
  __shared__ curandState state;
  if (threadIdx.x == 0) {
    curand_init(1234, blockIdx.x, 0, &state);
  }
  __syncthreads();

  for (int i = idx; i < numel; i += stride) {
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
