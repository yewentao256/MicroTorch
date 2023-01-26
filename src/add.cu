#include <math.h>  // function to add the elements of two arrays
#include "cuda_impl.hpp"
#include <iostream>

namespace tinytorch {
// CUDA Kernel function to add the elements of two arrays on the GPU
__global__ void add(int n, float *a, float *b, float *out){ 
  int index = blockIdx.x * blockDim.x + threadIdx.x; 
  int stride = blockDim.x * gridDim.x; 
  for (int i = index; i < n; i += stride) {
    out[i] = a[i] + b[i];
  }
}

Tensor add_cuda_impl(Tensor a, Tensor b) {
    Tensor result(a.size());

    float* a_ptr = a.data();
    float* b_ptr = b.data();
    float* out_ptr = result.data();
    std::size_t size = sizeof(float) * result.size();

    cudaMallocManaged(&a_ptr, size);
    cudaMallocManaged(&b_ptr, size);
    cudaMallocManaged(&out_ptr, size);

    int blockSize = 256;
    int numBlocks = (result.size() + blockSize - 1) / blockSize;  // Ceilling
    add<<<numBlocks, blockSize>>>(result.size(), a_ptr, b_ptr, out_ptr);

    cudaDeviceSynchronize();

    cudaFree(a_ptr);
    cudaFree(b_ptr);
    return result;
}

}  // namespace tinytorch
