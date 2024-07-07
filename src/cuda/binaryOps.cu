/**
 * Copyright (c) 2022-2024 yewentao256
 * Licensed under the MIT License.
 */
#include "cuda.hpp"
#include "functors.hpp"
#include "loops.cuh"
#include "ops.hpp"

namespace microtorch {

template <>
void add_impl<Cuda>(TensorIterator &iter) {
  gpu_kernel(iter, binaryFunctor::Add());
}

template <>
void sub_impl<Cuda>(TensorIterator &iter) {
  gpu_kernel(iter, binaryFunctor::Sub());
}

template <>
void mul_impl<Cuda>(TensorIterator &iter) {
  gpu_kernel(iter, binaryFunctor::Mul());
}

template <>
void div_impl<Cuda>(TensorIterator &iter) {
  gpu_kernel(iter, binaryFunctor::Div());
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
