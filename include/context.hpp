#pragma once

#include <map>
#include "tensor.hpp"

namespace microtorch {

// placeholder for template
class Host {};
class Cuda {};

struct Context {
  std::map<std::string, Tensor> data;
  Device device;
  Context(const Device& device = Device("cpu")) : device(device) {}
};

}  // namespace microtorch