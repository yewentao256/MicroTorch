#include "ops.hpp"

namespace microtorch {

__global__ void fill(size_t n, float *self, float value) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = index; i < n; i += stride) {
    self[i] = value;
  }
}

template <>
void fill_impl<Cuda>(Tensor &self, const data_t value) {
  float *self_ptr = self.data_ptr();

  size_t blockSize = 256;
  size_t numBlocks = (self.numel() + blockSize - 1) / blockSize;  // Ceilling
  fill<<<numBlocks, blockSize>>>(self.numel(), self_ptr, value);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
  }
}

__global__ void clone(size_t n, const float *a, float *out) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = index; i < n; i += stride) {
    out[i] = a[i];
  }
}

template <>
void clone_impl<Cuda>(const Tensor &a, Tensor &out) {
  float *a_ptr = a.data_ptr();
  float *out_ptr = out.data_ptr();

  size_t blockSize = 256;
  size_t numBlocks = (a.numel() + blockSize - 1) / blockSize;  // Ceilling
  clone<<<numBlocks, blockSize>>>(a.numel(), a_ptr, out_ptr);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
  }
}

__global__ void clone_backward(size_t n, const float *grad_output_ptr,
                               float *grad_input_ptr) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = index; i < n; i += stride) {
    grad_input_ptr[i] = grad_output_ptr[i];
  }
}

template <>
void clone_backward_impl<Cuda>(const Tensor &grad_output, Tensor &grad_input) {
  float *grad_output_ptr = grad_output.data_ptr();
  float *grad_input_ptr = grad_input.data_ptr();

  size_t blockSize = 256;
  size_t numBlocks =
      (grad_output.numel() + blockSize - 1) / blockSize;  // Ceilling
  clone_backward<<<numBlocks, blockSize>>>(grad_output.numel(),
                                            grad_output_ptr, grad_input_ptr);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
  }
}

}  // namespace microtorch
