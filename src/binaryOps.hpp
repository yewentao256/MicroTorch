#pragma once

#include "context.hpp"
#include "engine.hpp"
#include "graph.hpp"
#include "tensor.hpp"
#include "tensorFactories.hpp"

namespace tinytorch {

inline void add_out(const Tensor& a, const Tensor& b, Tensor& out) {
  OpNode("add").ins({a, b}).outs({out}).apply();
}
inline void sub_out(const Tensor& a, const Tensor& b, Tensor& out) {
  OpNode("sub").ins({a, b}).outs({out}).apply();
}
inline void mul_out(const Tensor& a, const Tensor& b, Tensor& out) {
  OpNode("mul").ins({a, b}).outs({out}).apply();
}

// Internal implementation of forward/backward
// Should NOT be called by the user
template <typename Device>
void add_impl(Tensor& a, Tensor& b, Tensor& out);
template <typename Device>
void add_backward_impl(Tensor& grad_output, Tensor& grad_input_1,
                       Tensor& grad_input_2);

template <typename Device>
void sub_impl(Tensor& a, Tensor& b, Tensor& out);
template <typename Device>
void sub_backward_impl(Tensor& grad_output, Tensor& grad_input_1,
                       Tensor& grad_input_2);

template <typename Device>
void mul_impl(Tensor& a, Tensor& b, Tensor& out);
template <typename Device>
void mul_backward_impl(Tensor& grad_output, Tensor& grad_input_1,
                       Tensor& grad_input_2, Tensor& a, Tensor& b);

}  // namespace tinytorch
