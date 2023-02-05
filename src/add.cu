#include <math.h>  // function to add the elements of two arrays

#include <iostream>

#include "cuda_impl.hpp"

namespace tinytorch {

// CUDA Kernel function to add the elements of two arrays on the GPU
__global__ void add(size_t n, float *a, float *b, float *out) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = index; i < n; i += stride) {
    out[i] = a[i] + b[i];
  }
}

Tensor add_cuda_impl(Tensor a, Tensor b) {
  Tensor result(a.size());
  float *a_ptr = a.data_ptr();
  float *b_ptr = b.data_ptr();
  float *out_ptr = result.data_ptr();
  size_t size = sizeof(float) * result.size();

  float *a_ptr_cuda, *b_ptr_cuda, *out_ptr_cuda;
  cudaMalloc(&a_ptr_cuda, size);
  cudaMalloc(&b_ptr_cuda, size);
  cudaMalloc(&out_ptr_cuda, size);

  // Copy data from host arrays to device
  cudaMemcpy(a_ptr_cuda, a_ptr, size, cudaMemcpyHostToDevice);
  cudaMemcpy(b_ptr_cuda, b_ptr, size, cudaMemcpyHostToDevice);

  size_t blockSize = 256;
  size_t numBlocks = (result.size() + blockSize - 1) / blockSize;  // Ceilling
  add<<<numBlocks, blockSize>>>(result.size(), a_ptr_cuda, b_ptr_cuda,
                                out_ptr_cuda);

  cudaMemcpy(out_ptr, out_ptr_cuda, size, cudaMemcpyDeviceToHost);

  cudaFree(a_ptr_cuda);
  cudaFree(b_ptr_cuda);
  cudaFree(out_ptr_cuda);
  return result;
}

__global__ void add_backward(size_t n, float *grad, float *result_a,
                             float *result_b) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = index; i < n; i += stride) {
    // y = a + b, y'(a) = 1 * grad
    result_a[i] = grad[i];
    result_b[i] = grad[i];
  }
}

std::vector<Tensor> add_backward_cuda_impl(Tensor grad) {
  Tensor result_a(grad.size());
  Tensor result_b(grad.size());
  float *grad_ptr = grad.data_ptr();
  float *result_a_ptr = result_a.data_ptr();
  float *result_b_ptr = result_b.data_ptr();
  size_t size = sizeof(float) * grad.size();

  float *grad_ptr_cuda, *result_a_ptr_cuda, *result_b_ptr_cuda;
  cudaMalloc(&grad_ptr_cuda, size);
  cudaMalloc(&result_a_ptr_cuda, size);
  cudaMalloc(&result_b_ptr_cuda, size);

  cudaMemcpy(grad_ptr_cuda, grad_ptr, size, cudaMemcpyHostToDevice);

  size_t blockSize = 256;
  size_t numBlocks = (grad.size() + blockSize - 1) / blockSize;  // Ceilling
  add<<<numBlocks, blockSize>>>(grad.size(), grad_ptr_cuda, result_a_ptr_cuda,
                                result_b_ptr_cuda);

  cudaMemcpy(result_a_ptr, result_a_ptr_cuda, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(result_b_ptr, result_b_ptr_cuda, size, cudaMemcpyDeviceToHost);

  cudaFree(grad_ptr_cuda);
  cudaFree(result_a_ptr_cuda);
  cudaFree(result_b_ptr_cuda);
  return {result_a, result_b};
}

}  // namespace tinytorch
