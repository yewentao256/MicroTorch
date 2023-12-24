/**
 * Copyright (c) 2022-2023 yewentao
 * Licensed under the MIT License.
 */

#pragma once

namespace microtorch {
namespace internal {

template <typename T>
__host__ __device__ bool isAligned(T* pointer, size_t alignment) {
  return reinterpret_cast<uintptr_t>(pointer) % alignment == 0;
}

template <typename T>
struct LoadImpl {
  __device__ __host__ static T apply(const void* src) {
    unsigned char* bytePtr = (unsigned char*)src;
    printf("is aligned: %d, pointer: %p, value: 0x%02x%02x%02x%02x\n",
           isAligned(src, 4), (void*)src, bytePtr[0], bytePtr[1], bytePtr[2],
           bytePtr[3]);
    auto r = *reinterpret_cast<const T*>(src);
    printf("r: %f\n", r);
    return r;
  }
};

}  // namespace internal

template <typename T>
__device__ __host__ T load(const void* src) {
  return internal::LoadImpl<T>::apply(src);
}
}  // namespace microtorch
