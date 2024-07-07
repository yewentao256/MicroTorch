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

// Internal implementation of forward/backward
// Should NOT be called by the user
template <typename Device>
void add_impl(TensorIterator& iter);

template <typename Device>
void sub_impl(TensorIterator& iter);

template <typename Device>
void mul_impl(TensorIterator& iter);

template <typename Device>
void div_impl(TensorIterator& iter);

template <typename Device>
void eq_impl(const Tensor& a, const Tensor& b, Tensor& out,
             const float epsilon = 1e-5);

}  // namespace microtorch
