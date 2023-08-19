#pragma once
#include "graph.hpp"
#include "ops.hpp"
#include "tensor.hpp"

namespace tinytorch {

void backward(Tensor loss);

}  // namespace tinytorch