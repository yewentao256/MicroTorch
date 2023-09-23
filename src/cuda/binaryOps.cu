#include "cuda.hpp"
#include "ops.hpp"

namespace microtorch {

__global__ void add_kernel(size_t n, float *a, float *b, float *out) {
  // one dimension layout
  // blockDim.x: block size (threads per block)
  // gridDim.x: grid size (blocks per grid)
  // blockIdx.x: current block index in grid
  // threadIdx.x: current thread index in block.
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = index; i < n; i += stride) {
    out[i] = a[i] + b[i];
  }
}

template <>
void add_impl<Cuda>(Tensor &a, Tensor &b, Tensor &out) {
  size_t blocks_per_grid = get_blocks_per_grid(out.numel());
  add_kernel<<<blocks_per_grid, ThreadsPerBlock>>>(
      out.numel(), a.data_ptr(), b.data_ptr(), out.data_ptr());
  CUDA_ERROR_CHECK();
}

__global__ void add_backward_kernel(size_t n, float *grad_output,
                                    float *grad_input_1, float *grad_input_2) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = index; i < n; i += stride) {
    // y = a + b, y'(a) = 1 * grad
    grad_input_1[i] = grad_output[i];
    grad_input_2[i] = grad_output[i];
  }
}

template <>
void add_backward_impl<Cuda>(Tensor &grad_output, Tensor &grad_input_1,
                             Tensor &grad_input_2) {
  size_t blocks_per_grid = get_blocks_per_grid(grad_output.numel());
  add_backward_kernel<<<blocks_per_grid, ThreadsPerBlock>>>(
      grad_output.numel(), grad_output.data_ptr(), grad_input_1.data_ptr(),
      grad_input_2.data_ptr());
  CUDA_ERROR_CHECK();
}

__global__ void sub_kernel(size_t n, float *a, float *b, float *out) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = index; i < n; i += stride) {
    out[i] = a[i] - b[i];
  }
}

template <>
void sub_impl<Cuda>(Tensor &a, Tensor &b, Tensor &out) {
  size_t blocks_per_grid = get_blocks_per_grid(out.numel());
  sub_kernel<<<blocks_per_grid, ThreadsPerBlock>>>(
      out.numel(), a.data_ptr(), b.data_ptr(), out.data_ptr());
  CUDA_ERROR_CHECK();
}

__global__ void sub_backward_kernel(size_t n, float *grad_output,
                                    float *grad_input_1, float *grad_input_2) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = index; i < n; i += stride) {
    // y = a - b, y'(a) = 1 * grad, y'(b) = -1 * grad
    grad_input_1[i] = grad_output[i];
    grad_input_2[i] = -grad_output[i];
  }
}

template <>
void sub_backward_impl<Cuda>(Tensor &grad_output, Tensor &grad_input_1,
                             Tensor &grad_input_2) {
  size_t blocks_per_grid = get_blocks_per_grid(grad_output.numel());
  sub_backward_kernel<<<blocks_per_grid, ThreadsPerBlock>>>(
      grad_output.numel(), grad_output.data_ptr(), grad_input_1.data_ptr(),
      grad_input_2.data_ptr());
  CUDA_ERROR_CHECK();
}

__global__ void mul_kernel(size_t n, float *a, float *b, float *out) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = index; i < n; i += stride) {
    out[i] = a[i] * b[i];
  }
}

template <>
void mul_impl<Cuda>(Tensor &a, Tensor &b, Tensor &out) {
  size_t blocks_per_grid = get_blocks_per_grid(out.numel());
  mul_kernel<<<blocks_per_grid, ThreadsPerBlock>>>(
      out.numel(), a.data_ptr(), b.data_ptr(), out.data_ptr());
  CUDA_ERROR_CHECK();
}

__global__ void mul_backward_kernel(size_t n, float *grad_output,
                                    float *grad_input_1, float *grad_input_2,
                                    float *a, float *b) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = index; i < n; i += stride) {
    // y = a * b, y'(a) = b * grad, y'(b) = a * grad
    grad_input_1[i] = b[i] * grad_output[i];
    grad_input_2[i] = a[i] * grad_output[i];
  }
}

template <>
void mul_backward_impl<Cuda>(Tensor &grad_output, Tensor &grad_input_1,
                             Tensor &grad_input_2, Tensor &a, Tensor &b) {
  size_t blocks_per_grid = get_blocks_per_grid(grad_output.numel());
  mul_backward_kernel<<<blocks_per_grid, ThreadsPerBlock>>>(
      grad_output.numel(), grad_output.data_ptr(), grad_input_1.data_ptr(),
      grad_input_2.data_ptr(), a.data_ptr(), b.data_ptr());
  CUDA_ERROR_CHECK();
}

__global__ void equal_kernel(size_t n, float *a, float *b, float *out,
                             const float epsilon) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = index; i < n; i += stride) {
    out[i] = fabsf(a[i] - b[i]) < epsilon;
  }
}

template <>
void equal_impl<Cuda>(const Tensor &a, const Tensor &b, Tensor &out,
                      const float epsilon) {
  size_t blocks_per_grid = get_blocks_per_grid(out.numel());
  equal_kernel<<<blocks_per_grid, ThreadsPerBlock>>>(
      out.numel(), a.data_ptr(), b.data_ptr(), out.data_ptr(), epsilon);
  CUDA_ERROR_CHECK();
}

}  // namespace microtorch
