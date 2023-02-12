#pragma once

#include <map>
#include "tensor.hpp"

namespace tinytorch {

// placeholder for template
class Host {};
class Cuda {};

struct Context {
  std::map<std::string, Tensor> data;
  std::map<std::string, int> data_int;
  std::string arch;
};

}  // namespace tinytorch