/**
 * Copyright (c) 2022-2023 yewentao
 * Licensed under the MIT License.
 */
#include "binaryOps.hpp"

#include "functors.hpp"
#include "loops.hpp"
#include "tensorIterator.hpp"

namespace microtorch {

template <>
void add_impl<Host>(TensorIterator& iter) {
  cpu_kernel(iter, binaryFunctor::Add());
}

template <>
void sub_impl<Host>(TensorIterator& iter) {
  cpu_kernel(iter, binaryFunctor::Sub());
}

template <>
void mul_impl<Host>(TensorIterator& iter) {
  cpu_kernel(iter, binaryFunctor::Mul());
}

template <>
void div_impl<Host>(TensorIterator& iter) {
    cpu_kernel(iter, binaryFunctor::Div());
}

template <>
void eq_impl<Host>(const Tensor& a, const Tensor& b, Tensor& out,
                   const float epsilon) {
  auto a_ptr = a.data_ptr();
  auto b_ptr = b.data_ptr();
  auto out_ptr = out.data_ptr();
  for (int64_t i = 0; i < out.numel(); i++) {
    out_ptr[i] = std::abs(a_ptr[i] - b_ptr[i]) < epsilon;
  }
}

}  // namespace microtorch
