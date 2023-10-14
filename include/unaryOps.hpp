#pragma once

#include "context.hpp"
#include "engine.hpp"
#include "graph.hpp"
#include "tensor.hpp"
#include "tensorFactories.hpp"

namespace microtorch {

inline Tensor square(Tensor &a) {
  return a * a;
}

}  // namespace microtorch
