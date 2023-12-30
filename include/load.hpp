/**
 * Copyright (c) 2022-2023 yewentao
 * Licensed under the MIT License.
 */

#pragma once
#include "macros.hpp"

namespace microtorch {
namespace internal {

template <typename T>
struct LoadImpl {
  HOST_DEVICE static T apply(const void* src) {
    return *reinterpret_cast<const T*>(src);
  }
};

}  // namespace internal

template <typename T>
HOST_DEVICE T load(const void* src) {
  return internal::LoadImpl<T>::apply(src);
}
}  // namespace microtorch
