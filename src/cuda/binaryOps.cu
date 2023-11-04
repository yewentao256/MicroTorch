/**
 * Copyright (c) 2022-2023 yewentao
 * Licensed under the MIT License.
 */
#include "cuda.hpp"
#include "ops.hpp"

namespace microtorch {

__global__ void add_kernel(int64_t n, float *a, float *b, float *out) {
  // one dimension layout
  // blockDim.x: block size (threads per block)
  // gridDim.x: grid size (blocks per grid)
  // blockIdx.x: current block index in grid
  // threadIdx.x: current thread index in block.
  // e.g.: blockDim.x = 4, gridDim.x = 4, we have 16 threads in total
  // index ranging from 0, 1, 2 ... to 13, 14, 15, stride = 16
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;
  for (int64_t i = index; i < n; i += stride) {
    out[i] = a[i] + b[i];
  }
}

template <>
void add_impl<Cuda>(const Tensor &a, const Tensor &b, Tensor &out) {
  int64_t blocks_per_grid = get_blocks_per_grid(out.numel());
  add_kernel<<<blocks_per_grid, ThreadsPerBlock>>>(
      out.numel(), a.data_ptr(), b.data_ptr(), out.data_ptr());
  CUDA_ERROR_CHECK();
}

__global__ void add_backward_kernel(int64_t n, float *grad_output,
                                    float *grad_input_1, float *grad_input_2) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    // y = a + b, y'(a) = 1 * grad
    grad_input_1[i] = grad_output[i];
    grad_input_2[i] = grad_output[i];
  }
}

template <>
void add_backward_impl<Cuda>(Tensor &grad_output, Tensor &grad_input_1,
                             Tensor &grad_input_2) {
  int64_t blocks_per_grid = get_blocks_per_grid(grad_output.numel());
  add_backward_kernel<<<blocks_per_grid, ThreadsPerBlock>>>(
      grad_output.numel(), grad_output.data_ptr(), grad_input_1.data_ptr(),
      grad_input_2.data_ptr());
  CUDA_ERROR_CHECK();
}

__global__ void sub_kernel(int64_t n, float *a, float *b, float *out) {
  // nvcc will optimize the constant value blockDim.x * gridDim.x
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    out[i] = a[i] - b[i];
  }
}

template <>
void sub_impl<Cuda>(const Tensor &a, const Tensor &b, Tensor &out) {
  int64_t blocks_per_grid = get_blocks_per_grid(out.numel());
  sub_kernel<<<blocks_per_grid, ThreadsPerBlock>>>(
      out.numel(), a.data_ptr(), b.data_ptr(), out.data_ptr());
  CUDA_ERROR_CHECK();
}

__global__ void sub_backward_kernel(int64_t n, float *grad_output,
                                    float *grad_input_1, float *grad_input_2) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    // y = a - b, y'(a) = 1 * grad, y'(b) = -1 * grad
    grad_input_1[i] = grad_output[i];
    grad_input_2[i] = -grad_output[i];
  }
}

template <>
void sub_backward_impl<Cuda>(Tensor &grad_output, Tensor &grad_input_1,
                             Tensor &grad_input_2) {
  int64_t blocks_per_grid = get_blocks_per_grid(grad_output.numel());
  sub_backward_kernel<<<blocks_per_grid, ThreadsPerBlock>>>(
      grad_output.numel(), grad_output.data_ptr(), grad_input_1.data_ptr(),
      grad_input_2.data_ptr());
  CUDA_ERROR_CHECK();
}

__global__ void mul_kernel(int64_t n, float *a, float *b, float *out) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    out[i] = a[i] * b[i];
  }
}

template <>
void mul_impl<Cuda>(const Tensor &a, const Tensor &b, Tensor &out) {
  int64_t blocks_per_grid = get_blocks_per_grid(out.numel());
  mul_kernel<<<blocks_per_grid, ThreadsPerBlock>>>(
      out.numel(), a.data_ptr(), b.data_ptr(), out.data_ptr());
  CUDA_ERROR_CHECK();
}

__global__ void mul_backward_kernel(int64_t n, float *grad_output,
                                    float *grad_input_1, float *grad_input_2,
                                    float *a, float *b) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    // y = a * b, y'(a) = b * grad, y'(b) = a * grad
    grad_input_1[i] = b[i] * grad_output[i];
    grad_input_2[i] = a[i] * grad_output[i];
  }
}

template <>
void mul_backward_impl<Cuda>(const Tensor &grad_output, Tensor &grad_input_1,
                             Tensor &grad_input_2, const Tensor &a,
                             const Tensor &b) {
  int64_t blocks_per_grid = get_blocks_per_grid(grad_output.numel());
  mul_backward_kernel<<<blocks_per_grid, ThreadsPerBlock>>>(
      grad_output.numel(), grad_output.data_ptr(), grad_input_1.data_ptr(),
      grad_input_2.data_ptr(), a.data_ptr(), b.data_ptr());
  CUDA_ERROR_CHECK();
}

__global__ void mul_scalar_kernel(int64_t n, float *a, float b, float *out) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    out[i] = a[i] * b;
  }
}

template <>
void mul_scalar_impl<Cuda>(const Tensor &a, const float b, Tensor &out) {
  int64_t blocks_per_grid = get_blocks_per_grid(out.numel());
  mul_scalar_kernel<<<blocks_per_grid, ThreadsPerBlock>>>(
      out.numel(), a.data_ptr(), b, out.data_ptr());
  CUDA_ERROR_CHECK();
}

__global__ void mul_scalar_backward_kernel(int64_t n, float *grad_output,
                                           float *grad_input, float b) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    grad_input[i] = b * grad_output[i];
  }
}

template <>
void mul_scalar_backward_impl<Cuda>(const Tensor &grad_output,
                                    Tensor &grad_input, const float b) {
  int64_t blocks_per_grid = get_blocks_per_grid(grad_output.numel());
  mul_scalar_backward_kernel<<<blocks_per_grid, ThreadsPerBlock>>>(
      grad_output.numel(), grad_output.data_ptr(), grad_input.data_ptr(), b);
  CUDA_ERROR_CHECK();
}

__global__ void div_kernel(int64_t n, float *a, float *b, float *out) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    out[i] = a[i] / b[i];
  }
}

template <>
void div_impl<Cuda>(const Tensor &a, const Tensor &b, Tensor &out) {
  int64_t blocks_per_grid = get_blocks_per_grid(out.numel());
  div_kernel<<<blocks_per_grid, ThreadsPerBlock>>>(
      out.numel(), a.data_ptr(), b.data_ptr(), out.data_ptr());
  CUDA_ERROR_CHECK();
}

__global__ void div_backward_kernel(int64_t n, float *grad_output,
                                    float *grad_input_1, float *grad_input_2,
                                    float *a, float *b) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    // y = a / b, y'(a) = 1 / b * grad, y'(b) = -a * (1/b)^-2 grad
    float reciprocal_b = 1 / b[i];
    grad_input_1[i] = 1 / b[i] * grad_output[i];
    grad_input_2[i] = -1 * a[i] * reciprocal_b * reciprocal_b * grad_output[i];
  }
}

template <>
void div_backward_impl<Cuda>(const Tensor &grad_output, Tensor &grad_input_1,
                             Tensor &grad_input_2, const Tensor &a,
                             const Tensor &b) {
  int64_t blocks_per_grid = get_blocks_per_grid(grad_output.numel());
  div_backward_kernel<<<blocks_per_grid, ThreadsPerBlock>>>(
      grad_output.numel(), grad_output.data_ptr(), grad_input_1.data_ptr(),
      grad_input_2.data_ptr(), a.data_ptr(), b.data_ptr());
  CUDA_ERROR_CHECK();
}

__global__ void equal_kernel(int64_t n, float *a, float *b, float *out,
                             const float epsilon) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    out[i] = fabsf(a[i] - b[i]) < epsilon;
  }
}

template <>
void eq_impl<Cuda>(const Tensor &a, const Tensor &b, Tensor &out,
                   const float epsilon) {
  int64_t blocks_per_grid = get_blocks_per_grid(out.numel());
  equal_kernel<<<blocks_per_grid, ThreadsPerBlock>>>(
      out.numel(), a.data_ptr(), b.data_ptr(), out.data_ptr(), epsilon);
  CUDA_ERROR_CHECK();
}

}  // namespace microtorch
