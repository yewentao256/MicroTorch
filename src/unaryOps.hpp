#pragma once

#include "context.hpp"
#include "tensor.hpp"
#include "graph.hpp"
#include "engine.hpp"
#include "tensorFactories.hpp"

namespace tinytorch {

template <typename Device>
void square_impl(Tensor& a, Tensor& out);
std::vector<Tensor> square_backward_impl(Tensor a, Tensor& dy);

inline Tensor square(Tensor& a) {
  Tensor out = zeros(a.shape(), a.device());
  OpNode("square").ins({a}).outs({out}).apply();
  return out;
}

}  // namespace tinytorch
