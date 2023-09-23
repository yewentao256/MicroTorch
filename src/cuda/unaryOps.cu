#include "cuda.hpp"
#include "ops.hpp"

namespace microtorch {

__global__ void square_kernel(size_t n, float *a, float *out) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = index; i < n; i += stride) {
    out[i] = a[i] * a[i];
  }
}

template <>
void square_impl<Cuda>(Tensor &a, Tensor &out) {
  size_t blocks_per_grid = get_blocks_per_grid(a.numel());
  square_kernel<<<blocks_per_grid, ThreadsPerBlock>>>(a.numel(), a.data_ptr(),
                                                      out.data_ptr());
  CUDA_ERROR_CHECK();
}

__global__ void square_backward_kernel(size_t n, float *grad_output_ptr,
                                       float *grad_input_ptr,
                                       float *input_ptr) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = index; i < n; i += stride) {
    // y = a^2, y'(a) = 2 * a * grad
    grad_input_ptr[i] = 2 * input_ptr[i] * grad_output_ptr[i];
  }
}

template <>
void square_backward_impl<Cuda>(Tensor &grad_output, Tensor &grad_input,
                                Tensor &input) {
  size_t blocks_per_grid = get_blocks_per_grid(grad_output.numel());
  square_backward_kernel<<<blocks_per_grid, ThreadsPerBlock>>>(
      grad_output.numel(), grad_output.data_ptr(), grad_input.data_ptr(),
      input.data_ptr());
  CUDA_ERROR_CHECK();
}

}  // namespace microtorch
