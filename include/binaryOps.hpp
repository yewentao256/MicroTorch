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
void mul_backward_impl(Tensor& grad_output, Tensor& grad_input_1,
                       Tensor& grad_input_2, Tensor& a, Tensor& b);

template <typename Device>
void equal_impl(const Tensor& a, const Tensor& b, Tensor& out, const float epsilon = 1e-6);
template <>
void equal_impl<Host>(const Tensor& a, const Tensor& b, Tensor& out, const float epsilon);
template <>
void equal_impl<Cuda>(const Tensor& a, const Tensor& b, Tensor& out, const float epsilon);

// user interface
inline void equal_out(const Tensor& a, const Tensor& b, Tensor& out) {
  DISPATCH_OP(equal_impl, a.device(), a, b, out);
}
}  // namespace microtorch
