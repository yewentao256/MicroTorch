#pragma once

#include "context.hpp"
#include "engine.hpp"
#include "graph.hpp"
#include "tensor.hpp"
#include "tensorFactories.hpp"

namespace microtorch {

template <typename Device>
void square_impl(Tensor &a, Tensor &out);
template <typename Device>
void square_backward_impl(Tensor &grad_output, Tensor &grad_input,
                          Tensor &input);

inline Tensor square(Tensor &a) {
  Tensor out = zeros(a.shape(), a.device());
  OpNode("square").ins({a}).outs({out}).apply();
  return out;
}

}  // namespace microtorch
