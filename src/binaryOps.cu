#include <math.h>  // function to add the elements of two arrays

#include "ops.hpp"

namespace tinytorch {

__global__ void add(size_t n, float *a, float *b, float *out) {
  // one dimension layout
  // blockDim.x: block size
  // gridDim.x: grid size (how many blocks)
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
  float *a_ptr = a.data_ptr();
  float *b_ptr = b.data_ptr();
  float *out_ptr = out.data_ptr();

  size_t blockSize = 256;
  size_t numBlocks = (out.numel() + blockSize - 1) / blockSize;  // Ceilling
  add<<<numBlocks, blockSize>>>(out.numel(), a_ptr, b_ptr, out_ptr);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
  }
}

__global__ void add_backward(size_t n, float *grad_output, float *grad_input_1,
                             float *grad_input_2) {
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
  float *grad_output_ptr = grad_output.data_ptr();
  float *grad_input_1_ptr = grad_input_1.data_ptr();
  float *grad_input_2_ptr = grad_input_2.data_ptr();

  size_t blockSize = 256;
  size_t numBlocks =
      (grad_output.numel() + blockSize - 1) / blockSize;  // Ceilling
  add_backward<<<numBlocks, blockSize>>>(grad_output.numel(), grad_output_ptr,
                                         grad_input_1_ptr, grad_input_2_ptr);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
  }
}

__global__ void sub(size_t n, float *a, float *b, float *out) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = index; i < n; i += stride) {
    out[i] = a[i] - b[i];
  }
}

template <>
void sub_impl<Cuda>(Tensor &a, Tensor &b, Tensor &out) {
  float *a_ptr = a.data_ptr();
  float *b_ptr = b.data_ptr();
  float *out_ptr = out.data_ptr();

  size_t blockSize = 256;
  size_t numBlocks = (out.numel() + blockSize - 1) / blockSize;  // Ceilling
  sub<<<numBlocks, blockSize>>>(out.numel(), a_ptr, b_ptr, out_ptr);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
  }
}

__global__ void sub_backward(size_t n, float *grad_output, float *grad_input_1,
                             float *grad_input_2) {
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
  float *grad_output_ptr = grad_output.data_ptr();
  float *grad_input_1_ptr = grad_input_1.data_ptr();
  float *grad_input_2_ptr = grad_input_2.data_ptr();

  size_t blockSize = 256;
  size_t numBlocks =
      (grad_output.numel() + blockSize - 1) / blockSize;  // Ceilling
  sub_backward<<<numBlocks, blockSize>>>(grad_output.numel(), grad_output_ptr,
                                         grad_input_1_ptr, grad_input_2_ptr);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
  }
}

__global__ void mul(size_t n, float *a, float *b, float *out) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = index; i < n; i += stride) {
    out[i] = a[i] * b[i];
  }
}

template <>
void mul_impl<Cuda>(Tensor &a, Tensor &b, Tensor &out) {
  float *a_ptr = a.data_ptr();
  float *b_ptr = b.data_ptr();
  float *out_ptr = out.data_ptr();

  size_t blockSize = 256;
  size_t numBlocks = (out.numel() + blockSize - 1) / blockSize;  // Ceilling
  mul<<<numBlocks, blockSize>>>(out.numel(), a_ptr, b_ptr, out_ptr);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
  }
}

__global__ void mul_backward(size_t n, float *grad_output, float *grad_input_1,
                             float *grad_input_2, float *a, float *b) {
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
  float *grad_output_ptr = grad_output.data_ptr();
  float *grad_input_1_ptr = grad_input_1.data_ptr();
  float *grad_input_2_ptr = grad_input_2.data_ptr();

  float *a_ptr = a.data_ptr();
  float *b_ptr = b.data_ptr();

  size_t blockSize = 256;
  size_t numBlocks =
      (grad_output.numel() + blockSize - 1) / blockSize;  // Ceilling
  mul_backward<<<numBlocks, blockSize>>>(grad_output.numel(), grad_output_ptr,
                                         grad_input_1_ptr, grad_input_2_ptr,
                                         a_ptr, b_ptr);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
  }
}

}  // namespace tinytorch
