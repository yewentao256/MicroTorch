#include <math.h>  // function to add the elements of two arrays

#include <iostream>

#include "ops.hpp"

namespace tinytorch {

// CUDA Kernel function to add the elements of two arrays on the GPU
__global__ void add(size_t n, float *a, float *b, float *out) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = index; i < n; i += stride) {
    out[i] = a[i] + b[i];
  }
}

template <>
void add_impl<Cuda>(Tensor &a, Tensor &b, Tensor &out) {
  float *a_ptr = a.data_ptr();
  float *b_ptr = b.data_ptr();
  float *out_ptr = out.data_ptr();

  size_t blockSize = 256;
  size_t numBlocks = (out.size() + blockSize - 1) / blockSize;  // Ceilling
  add<<<numBlocks, blockSize>>>(out.size(), a_ptr, b_ptr, out_ptr);
}

__global__ void add_backward(size_t n, float *dy, float *dx_1, float *dx_2) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = index; i < n; i += stride) {
    // y = a + b, y'(a) = 1 * grad
    dx_1[i] = dy[i];
    dx_2[i] = dy[i];
  }
}

template <>
void add_backward_impl<Cuda>(Tensor &dy, Tensor &dx_1, Tensor &dx_2) {
  float *dy_ptr = dy.data_ptr();
  float *dx_1_ptr = dx_1.data_ptr();
  float *dx_2_ptr = dx_2.data_ptr();

  size_t blockSize = 256;
  size_t numBlocks = (dy.size() + blockSize - 1) / blockSize;  // Ceilling
  add_backward<<<numBlocks, blockSize>>>(dy.size(), dy_ptr, dx_1_ptr, dx_2_ptr);
}

}  // namespace tinytorch
