/**
 * Copyright (c) 2022-2023 yewentao
 * Licensed under the MIT License.
 */
#pragma once

#ifdef USE_CUDA

#include <cstdint>
#define HOST_DEVICE __device__ __host__
#define INLINE __inline__

constexpr int MAX_DIMS = 8;
// constants from
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications
#if __CUDA_ARCH__ == 750
constexpr uint32_t CUDA_MAX_THREADS_PER_SM = 1024;
#elif __CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 870 || __CUDA_ARCH__ == 890
constexpr uint32_t CUDA_MAX_THREADS_PER_SM = 1536;
#else
constexpr uint32_t CUDA_MAX_THREADS_PER_SM = 2048;
#endif
constexpr uint32_t CUDA_MAX_THREADS_PER_BLOCK = 1024;
constexpr uint32_t CUDA_THREADS_PER_BLOCK_FALLBACK = 256;
#define MAX_THREADS_PER_BLOCK(val)               \
  (((val) <= CUDA_MAX_THREADS_PER_BLOCK) ? (val) \
                                         : CUDA_THREADS_PER_BLOCK_FALLBACK)
#define MIN_BLOCKS_PER_SM(threads_per_block, blocks_per_sm)            \
  ((((threads_per_block) * (blocks_per_sm) <= CUDA_MAX_THREADS_PER_SM) \
        ? (blocks_per_sm)                                              \
        : ((CUDA_MAX_THREADS_PER_SM + (threads_per_block)-1) /         \
           (threads_per_block))))

#else
#define HOST_DEVICE
#define INLINE inline
#endif

using data_t = float;
