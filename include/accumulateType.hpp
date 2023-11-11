/**
 * Copyright (c) 2022-2023 yewentao
 * Licensed under the MIT License.
 */
#pragma once
#include "device.hpp"
#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_fp16.h>
#endif

namespace microtorch {

template <typename T, DeviceType D>
struct AccumulateTypeDevice {};

template <>
struct AccumulateTypeDevice<float, DeviceType::CUDA> {
  using type = float;
};
template <>
struct AccumulateTypeDevice<float, DeviceType::CPU> {
  using type = double;
};

template <typename T, bool>
struct AccumulateType {};

template <typename T>
struct AccumulateType<T, false> {
  using type = typename AccumulateTypeDevice<T, DeviceType::CPU>::type;
};

template <typename T>
struct AccumulateType<T, true> {
  using type = typename AccumulateTypeDevice<T, DeviceType::CUDA>::type;
};

template <typename T, bool is_cuda>
using acc_type = typename AccumulateType<T, is_cuda>::type;

}  // namespace microtorch
