#pragma once

#include "context.hpp"
#include "tensor.hpp"
#include "graph.hpp"
#include "engine.hpp"
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
void add_backward_impl(Tensor& dy, Tensor& dx_1, Tensor& dx_2);

template <typename Device>
void sub_impl(Tensor& a, Tensor& b, Tensor& out);
std::vector<Tensor> sub_backward_impl(Tensor& dy);

template <typename Device>
void mult_impl(Tensor& a, Tensor& b, Tensor& out);
std::vector<Tensor> mult_backward_impl(Tensor a, Tensor b, Tensor& dy);

}  // namespace tinytorch
