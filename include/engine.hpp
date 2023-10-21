/**
 * Copyright (c) 2022-2023 yewentao
 * Licensed under the MIT License.
 */
#pragma once

#include "tensor.hpp"

namespace microtorch {

#ifdef USE_CUDA
#define DISPATCH_OP(func, device, ...) \
  if (device.is_cpu()) {               \
    func<Host>(__VA_ARGS__);           \
  } else {                             \
    func<Cuda>(__VA_ARGS__);           \
  }
#else
#define DISPATCH_OP(func, device, ...)                                   \
  if (device.is_cpu()) {                                                 \
    func<Host>(__VA_ARGS__);                                             \
  } else {                                                               \
    std::cout << "Not support device in host compile mode" << std::endl; \
  }
#endif

}  // namespace microtorch