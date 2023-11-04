/**
 * Copyright (c) 2022-2023 yewentao
 * Licensed under the MIT License.
 */
#pragma once

#include "context.hpp"
#include "engine.hpp"
#include "graph.hpp"
#include "tensor.hpp"
#include "tensorFactories.hpp"

namespace microtorch {

// Internal implementation of forward/backward
// Should NOT be called by the user
template <typename Device>
void add_impl(const Tensor& a, const Tensor& b, Tensor& out);
template <typename Device>
void add_backward_impl(Tensor& grad_output, Tensor& grad_input_1,
                       Tensor& grad_input_2);

template <typename Device>
void sub_impl(const Tensor& a, const Tensor& b, Tensor& out);
template <typename Device>
void sub_backward_impl(Tensor& grad_output, Tensor& grad_input_1,
                       Tensor& grad_input_2);

template <typename Device>
void mul_impl(const Tensor& a, const Tensor& b, Tensor& out);
template <typename Device>
void mul_backward_impl(const Tensor& grad_output, Tensor& grad_input_1,
                       Tensor& grad_input_2, const Tensor& a, const Tensor& b);

template <typename Device>
void mul_scalar_impl(const Tensor& a, const float b, Tensor& out);
template <typename Device>
void mul_scalar_backward_impl(const Tensor& grad_output, Tensor& grad_input,
                              const float b);

template <typename Device>
void div_impl(const Tensor& a, const Tensor& b, Tensor& out);
template <typename Device>
void div_backward_impl(const Tensor& grad_output, Tensor& grad_input_1,
                       Tensor& grad_input_2, const Tensor& a, const Tensor& b);

template <typename Device>
void eq_impl(const Tensor& a, const Tensor& b, Tensor& out,
             const float epsilon = 1e-5);

}  // namespace microtorch
