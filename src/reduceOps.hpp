#pragma once

#include "context.hpp"
#include "tensor.hpp"
#include "graph.hpp"
#include "engine.hpp"
#include "tensorFactories.hpp"

namespace tinytorch {

template <typename Device>
void sum_impl(Tensor& a, Tensor& out);
std::vector<Tensor> sum_backward_impl(Tensor& input, Tensor& grad_output);

inline Tensor sum(Tensor& a) {
  Tensor out = zeros(1, a.device());
  OpNode("sum").ins({a}).outs({out}).apply();
  return out;
}
}  // namespace tinytorch
