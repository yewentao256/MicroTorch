#pragma once

#include "context.hpp"
#include "tensor.hpp"
#include "graph.hpp"
#include "engine.hpp"
#include "tensorFactories.hpp"

namespace microtorch {

template <typename Device>
void sum_impl(Tensor& a, Tensor& out);

inline Tensor sum(Tensor& a) {
  Tensor out = zeros({1}, a.device());
  OpNode("sum").ins({a}).outs({out}).apply();
  return out;
}
}  // namespace microtorch
