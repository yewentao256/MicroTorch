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

constexpr size_t bitset_size = 64;

namespace microtorch {

template <typename Device>
void sum_impl(const Tensor& a, Tensor& out);
template <typename Device>
void sum_dim_impl(const Tensor& a, Tensor& out, IntArrayRef& dims,
                  bool keep_dim = false);

Tensor sum(const Tensor& a);
Tensor sum_dim(const Tensor& a, IntArrayRef dims, bool keep_dim = false);

}  // namespace microtorch
