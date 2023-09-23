#pragma once

#include <algorithm>
#include <vector>

namespace microtorch {

inline constexpr size_t ThreadsPerBlock = 256;
inline constexpr size_t MaxBlocksPerGrid = 1024;

inline size_t get_blocks_per_grid(size_t numel) {
  return std::min(MaxBlocksPerGrid,
                  (numel + ThreadsPerBlock - 1) / ThreadsPerBlock);
}

#ifdef USE_CUDA
#define CUDA_ERROR_CHECK()                                    \
  do {                                                        \
    cudaError_t err = cudaGetLastError();                     \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
    }                                                         \
  } while (0)
#else
#define CUDA_ERROR_CHECK() \
  do {                     \
  } while (0)
#endif

}  // namespace microtorch