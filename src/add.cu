#include <math.h>  // function to add the elements of two arrays
#include "cuda_impl.hpp"
#include <iostream>

namespace tinytorch {
// CUDA Kernel function to add the elements of two arrays on the GPU
__global__ void add(int n, Tensor *a, Tensor *b, Tensor *out){ 
  int index = blockIdx.x * blockDim.x + threadIdx.x; 
  int stride = blockDim.x * gridDim.x; 
  for (int i = index; i < n; i += stride) {
    (*out)[i] = (*a)[i] + (*b)[i];
  }
}

Tensor * add_cuda_impl(Tensor *a, Tensor *b) {
    Tensor result(a->size());
    Tensor *out = &result;

    cudaMallocManaged(&a, sizeof(*a));
    cudaMallocManaged(&b, sizeof(*b));
    cudaMallocManaged(&out, sizeof(*out));

    // Run kernel on 1M elements on the GPU
    int blockSize = 256;
    int numBlocks = (a->size() + blockSize - 1) / blockSize;  // Ceilling
    add<<<numBlocks, blockSize>>>(a->size(), a, b, out);
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    // Check for errors (all values should be 3.0f)
    cudaFree(a);
    cudaFree(b);
    return out;
}

}  // namespace tinytorch