/**
 * Copyright (c) 2022-2023 yewentao
 * Licensed under the MIT License.
 */
#pragma once

#include "context.hpp"
#include "engine.hpp"
#include "tensor.hpp"
#include "tensorIterator.hpp"

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

inline Tensor empty(IntArrayRef size, const std::string& device,
                    bool requires_grad = false) {
  Tensor t(size, device, requires_grad);
  return t;
}
inline Tensor zeros(IntArrayRef size, const std::string& device,
                    bool requires_grad = false) {
  Tensor t = empty(size, device, requires_grad);
  fill_scalar(t, 0);
  return t;
}
inline Tensor ones(IntArrayRef size, const std::string& device,
                   bool requires_grad) {
  Tensor t = empty(size, device, requires_grad);
  fill_scalar(t, 1);
  return t;
}

inline Tensor rand(IntArrayRef size, const std::string& device,
                   bool requires_grad = false) {
  Tensor t = empty(size, device, requires_grad);
  DISPATCH_OP(rand_impl, t.device(), t);
  return t;
}

template <typename Device>
void clone_impl(TensorIterator& iter);

}  // namespace microtorch
