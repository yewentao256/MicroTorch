/**
 * Copyright (c) 2022-2024 yewentao256
 * Licensed under the MIT License.
 */
#pragma once

#include <map>
#include "tensor.hpp"

namespace microtorch {

// placeholder for template
class Host {};
class Cuda {};

struct Context {
  // TODO: find a better way to realize this
  std::map<std::string, Tensor> data;
  std::map<std::string, data_t> data_scalar;
  std::map<std::string, int64_t> data_int;
  Device device;
  Context(const Device& device = Device("cpu")) : device(device) {}
};

}  // namespace microtorch