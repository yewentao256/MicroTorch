#pragma once

#include "context.hpp"
#include "engine.hpp"
#include "tensor.hpp"

namespace microtorch {

template <typename Device>
void fill_impl(Tensor& self, const data_t value);
template <>
void fill_impl<Host>(Tensor& self, const data_t value);
template <>
void fill_impl<Cuda>(Tensor& self, const data_t value);

template <typename Device>
void rand_impl(Tensor& self);
template <>
void rand_impl<Host>(Tensor& self);
template <>
void rand_impl<Cuda>(Tensor& self);

inline void fill_scalar(Tensor& self, const data_t value) {
  DISPATCH_OP(fill_impl, self.device(), self, value);
}

inline Tensor rand(ArrayRef size, const std::string& device,
                   bool requires_grad = false) {
  Tensor t(size, device, requires_grad);
  DISPATCH_OP(rand_impl, t.device(), t);
  return t;
}

Tensor zeros(ArrayRef size, const std::string& device,
             bool requires_grad = false);
Tensor ones(ArrayRef size, const std::string& device,
            bool requires_grad = false);

template <typename Device>
void clone_impl(const Tensor& a, Tensor& out);
template <typename Device>
void clone_backward_impl(const Tensor& grad_output, Tensor& grad_input);

inline void clone_out(const Tensor& a, Tensor& out) {
  OpNode("clone").ins({a}).outs({out}).apply();
}

}  // namespace microtorch
