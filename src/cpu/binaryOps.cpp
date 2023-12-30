/**
 * Copyright (c) 2022-2023 yewentao
 * Licensed under the MIT License.
 */
#include "binaryOps.hpp"

#include "loops.hpp"
#include "tensorIterator.hpp"
#include "functors.hpp"

namespace microtorch {

template <>
void add_impl<Host>(TensorIterator& iter) {
  cpu_kernel(iter, binaryFunctor::Add());
}

template <>
void add_backward_impl<Host>(Tensor& grad_output, Tensor& grad_input_1,
                             Tensor& grad_input_2) {
  auto grad_output_ptr = grad_output.data_ptr();
  auto grad_input_1_ptr = grad_input_1.data_ptr();
  auto grad_input_2_ptr = grad_input_2.data_ptr();
  for (int64_t i = 0; i < grad_output.numel(); i++) {
    // y = a + b, y'(a) = 1 * grad
    grad_input_1_ptr[i] = grad_output_ptr[i];
    grad_input_2_ptr[i] = grad_output_ptr[i];
  }
}

template <>
void sub_impl<Host>(const Tensor& a, const Tensor& b, Tensor& out) {
  auto out_ptr = out.data_ptr();
  auto a_ptr = a.data_ptr();
  auto b_ptr = b.data_ptr();
  for (int64_t i = 0; i < a.numel(); i++) {
    out_ptr[i] = a_ptr[i] - b_ptr[i];
  }
}

template <>
void sub_backward_impl<Host>(Tensor& grad_output, Tensor& grad_input_1,
                             Tensor& grad_input_2) {
  auto grad_output_ptr = grad_output.data_ptr();
  auto grad_input_1_ptr = grad_input_1.data_ptr();
  auto grad_input_2_ptr = grad_input_2.data_ptr();
  for (int64_t i = 0; i < grad_output.numel(); i++) {
    // y = a - b, y'(a) = 1 * grad, y'(b) = -1 * grad
    grad_input_1_ptr[i] = grad_output_ptr[i];
    grad_input_2_ptr[i] = -grad_output_ptr[i];
  }
}

template <>
void mul_impl<Host>(const Tensor& a, const Tensor& b, Tensor& out) {
  auto out_ptr = out.data_ptr();
  auto a_ptr = a.data_ptr();
  auto b_ptr = b.data_ptr();
  for (int64_t i = 0; i < a.numel(); i++) {
    out_ptr[i] = a_ptr[i] * b_ptr[i];
  }
}

template <>
void mul_backward_impl<Host>(const Tensor& grad_output, Tensor& grad_input_1,
                             Tensor& grad_input_2, const Tensor& a,
                             const Tensor& b) {
  auto grad_output_ptr = grad_output.data_ptr();
  auto grad_input_1_ptr = grad_input_1.data_ptr();
  auto grad_input_2_ptr = grad_input_2.data_ptr();
  auto a_ptr = a.data_ptr();
  auto b_ptr = b.data_ptr();

  for (int64_t i = 0; i < a.numel(); i++) {
    // y = a * b, y'(a) = b * grad, y'(b) = a * grad
    grad_input_1_ptr[i] = b_ptr[i] * grad_output_ptr[i];
    grad_input_2_ptr[i] = a_ptr[i] * grad_output_ptr[i];
  }
}

template <>
void mul_scalar_impl<Host>(const Tensor& a, const float b, Tensor& out) {
  auto out_ptr = out.data_ptr();
  auto a_ptr = a.data_ptr();
  for (int64_t i = 0; i < a.numel(); i++) {
    out_ptr[i] = a_ptr[i] * b;
  }
}

template <>
void mul_scalar_backward_impl<Host>(const Tensor& grad_output,
                                    Tensor& grad_input, const float b) {
  auto grad_output_ptr = grad_output.data_ptr();
  auto grad_input_ptr = grad_input.data_ptr();

  for (int64_t i = 0; i < grad_input.numel(); i++) {
    grad_input_ptr[i] = b * grad_output_ptr[i];
  }
}

template <>
void div_impl<Host>(const Tensor& a, const Tensor& b, Tensor& out) {
  auto out_ptr = out.data_ptr();
  auto a_ptr = a.data_ptr();
  auto b_ptr = b.data_ptr();
  for (int64_t i = 0; i < a.numel(); i++) {
    out_ptr[i] = a_ptr[i] / b_ptr[i];
  }
}

template <>
void div_backward_impl<Host>(const Tensor& grad_output, Tensor& grad_input_1,
                             Tensor& grad_input_2, const Tensor& a,
                             const Tensor& b) {
  auto grad_output_ptr = grad_output.data_ptr();
  auto grad_input_1_ptr = grad_input_1.data_ptr();
  auto grad_input_2_ptr = grad_input_2.data_ptr();
  auto a_ptr = a.data_ptr();
  auto b_ptr = b.data_ptr();

  for (int64_t i = 0; i < a.numel(); i++) {
    // y = a / b, y'(a) = 1 / b * grad, y'(b) = -a * (1/b)^-2 grad
    float reciprocal_b = 1 / b_ptr[i];
    grad_input_1_ptr[i] = 1 / b_ptr[i] * grad_output_ptr[i];
    grad_input_2_ptr[i] =
        -1 * a_ptr[i] * reciprocal_b * reciprocal_b * grad_output_ptr[i];
  }
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
