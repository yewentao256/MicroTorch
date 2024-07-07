/**
 * Copyright (c) 2022-2024 yewentao256
 * Licensed under the MIT License.
 */
#pragma once

#include "context.hpp"
#include "engine.hpp"
#include "graph.hpp"
#include "tensor.hpp"
#include "tensorFactories.hpp"
#include "tensorIterator.hpp"

namespace microtorch {

inline Tensor square(Tensor& a) { return a * a; }

template <typename Device>
void neg_impl(TensorIterator& iter);

}  // namespace microtorch
