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
void add_impl<Cuda>(Context &ctx, Tensor &a, Tensor &b, Tensor &out) {
  float *a_ptr = a.data_ptr();
  float *b_ptr = b.data_ptr();
  float *out_ptr = out.data_ptr();
  size_t size = sizeof(float) * out.size();

  float *a_ptr_cuda, *b_ptr_cuda, *out_ptr_cuda;
  cudaMalloc(&a_ptr_cuda, size);
  cudaMalloc(&b_ptr_cuda, size);
  cudaMalloc(&out_ptr_cuda, size);

  // Copy data from host arrays to device
  cudaMemcpy(a_ptr_cuda, a_ptr, size, cudaMemcpyHostToDevice);
  cudaMemcpy(b_ptr_cuda, b_ptr, size, cudaMemcpyHostToDevice);

  size_t blockSize = 256;
  size_t numBlocks = (out.size() + blockSize - 1) / blockSize;  // Ceilling
  add<<<numBlocks, blockSize>>>(out.size(), a_ptr_cuda, b_ptr_cuda,
                                out_ptr_cuda);

  cudaMemcpy(out_ptr, out_ptr_cuda, size, cudaMemcpyDeviceToHost);

  cudaFree(a_ptr_cuda);
  cudaFree(b_ptr_cuda);
  cudaFree(out_ptr_cuda);
}

/* template<>
void partialSum<Cuda>(Context& ctx, const DArrayLite& in,
        DArrayLite& out, size_t dim) {
    if (reduce::getReducePrim(in) == Prim::Int64) {
        dispatch_int_with<CudaPartialReduceAdaptor, Prim::Bool, int64_t>
            (in.elemType(), ctx, in, out, dim, reduce::Add(), Id(), Id());
    } else {
        dispatch_real<CudaPartialReduceAdaptor>
            (in.elemType(), ctx, in, out, dim, Add(), Id(), Id());
    }
}
 */
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
void add_backward_impl<Cuda>(Context &ctx, Tensor &dy, Tensor &dx_1,
                             Tensor &dx_2) {
  float *dy_ptr = dy.data_ptr();
  float *dx_1_ptr = dx_1.data_ptr();
  float *dx_2_ptr = dx_2.data_ptr();
  size_t size = sizeof(float) * dy.size();

  float *dy_ptr_cuda, *dx_1_ptr_cuda, *dx_2_ptr_cuda;
  cudaMalloc(&dy_ptr_cuda, size);
  cudaMalloc(&dx_1_ptr_cuda, size);
  cudaMalloc(&dx_2_ptr_cuda, size);

  cudaMemcpy(dy_ptr_cuda, dy_ptr, size, cudaMemcpyHostToDevice);

  size_t blockSize = 256;
  size_t numBlocks = (dy.size() + blockSize - 1) / blockSize;  // Ceilling
  add<<<numBlocks, blockSize>>>(dy.size(), dy_ptr_cuda, dx_1_ptr_cuda,
                                dx_2_ptr_cuda);

  cudaMemcpy(dx_1_ptr, dx_1_ptr_cuda, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(dx_2_ptr, dx_2_ptr_cuda, size, cudaMemcpyDeviceToHost);

  cudaFree(dy_ptr_cuda);
  cudaFree(dx_1_ptr_cuda);
  cudaFree(dx_2_ptr_cuda);
}

}  // namespace tinytorch
