#pragma once

#include <map>
#include "tensor.hpp"

namespace tinytorch {

// placeholder for template
class Host {};
class Cuda {};

struct Context {
  std::map<std::string, Tensor> data;
  Device device;
  Context(const Device& device = Device("cpu")) : device(device) {}
};

}  // namespace tinytorch