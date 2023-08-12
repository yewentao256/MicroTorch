#include "ops.hpp"

namespace tinytorch {

__global__ void square(size_t n, float *a, float *out) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = index; i < n; i += stride) {
    out[i] = a[i] * a[i];
  }
}

template <>
void square_impl<Cuda>(Tensor &a, Tensor &out) {
  float *a_ptr = a.data_ptr();
  float *out_ptr = out.data_ptr();

  size_t blockSize = 256;
  size_t numBlocks = (a.numel() + blockSize - 1) / blockSize;  // Ceilling
  square<<<numBlocks, blockSize>>>(a.numel(), a_ptr, out_ptr);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
  }
}

__global__ void square_backward(size_t n, float *grad_output_ptr,
                                float *grad_input_ptr, float *input_ptr) {
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
  float *grad_output_ptr = grad_output.data_ptr();
  float *grad_input_ptr = grad_input.data_ptr();
  float *input_ptr = input.data_ptr();

  size_t blockSize = 256;
  size_t numBlocks =
      (grad_output.numel() + blockSize - 1) / blockSize;  // Ceilling
  square_backward<<<numBlocks, blockSize>>>(grad_output.numel(), grad_output_ptr,
                                         grad_input_ptr, input_ptr);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
  }
}

}  // namespace tinytorch
