#include "cuda.hpp"
#include "ops.hpp"

namespace microtorch {

__global__ void sum_kernel(size_t n, float *a, float *out) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = index; i < n; i += stride) {
    // Note: multi-thread write here, we should use atomicAdd
    atomicAdd(&out[0], a[i]);
  }
}

template <>
void sum_impl<Cuda>(Tensor &a, Tensor &out) {
  size_t blocks_per_grid = get_blocks_per_grid(a.numel());
  sum_kernel<<<blocks_per_grid, ThreadsPerBlock>>>(a.numel(), a.data_ptr(),
                                                   out.data_ptr());
  CUDA_ERROR_CHECK();
}

}  // namespace microtorch
