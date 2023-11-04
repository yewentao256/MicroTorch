/**
 * Copyright (c) 2022-2023 yewentao
 * Licensed under the MIT License.
 */
#pragma once

#include "context.hpp"
#include "tensor.hpp"
#include "graph.hpp"
#include "engine.hpp"
#include "tensorFactories.hpp"

namespace microtorch {

template <typename Device>
void sum_impl(const Tensor& a, Tensor& out);

Tensor sum(const Tensor& a);
// Tensor sum(const Tensor& a, int64_t dim = -1, bool keep_dim = false);

}  // namespace microtorch
