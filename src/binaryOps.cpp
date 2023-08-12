
#include "binaryOps.hpp"

namespace tinytorch {

template <>
void add_impl<Host>(Tensor& a, Tensor& b, Tensor& out) {
  auto out_ptr = out.data_ptr();
  auto a_ptr = a.data_ptr();
  auto b_ptr = b.data_ptr();
  for (size_t i = 0; i < a.numel(); i++) {
    out_ptr[i] = a_ptr[i] + b_ptr[i];
  }
}

template <>
void add_backward_impl<Host>(Tensor& grad_output, Tensor& grad_input_1,
                             Tensor& grad_input_2) {
  auto grad_output_ptr = grad_output.data_ptr();
  auto grad_input_1_ptr = grad_input_1.data_ptr();
  auto grad_input_2_ptr = grad_input_2.data_ptr();
  for (size_t i = 0; i < grad_output.numel(); i++) {
    // y = a + b, y'(a) = 1 * grad
    grad_input_1_ptr[i] = grad_output_ptr[i];
    grad_input_2_ptr[i] = grad_output_ptr[i];
  }
}

template <>
void sub_impl<Host>(Tensor& a, Tensor& b, Tensor& out) {
  auto out_ptr = out.data_ptr();
  auto a_ptr = a.data_ptr();
  auto b_ptr = b.data_ptr();
  for (size_t i = 0; i < a.numel(); i++) {
    out_ptr[i] = a_ptr[i] - b_ptr[i];
  }
}

template <>
void sub_backward_impl<Host>(Tensor& grad_output, Tensor& grad_input_1,
                             Tensor& grad_input_2) {
  auto grad_output_ptr = grad_output.data_ptr();
  auto grad_input_1_ptr = grad_input_1.data_ptr();
  auto grad_input_2_ptr = grad_input_2.data_ptr();
  for (size_t i = 0; i < grad_output.numel(); i++) {
    // y = a - b, y'(a) = 1 * grad, y'(b) = -1 * grad
    grad_input_1_ptr[i] = grad_output_ptr[i];
    grad_input_2_ptr[i] = -grad_output_ptr[i];
  }
}

template <>
void mul_impl<Host>(Tensor& a, Tensor& b, Tensor& out) {
  auto out_ptr = out.data_ptr();
  auto a_ptr = a.data_ptr();
  auto b_ptr = b.data_ptr();
  for (size_t i = 0; i < a.numel(); i++) {
    out_ptr[i] = a_ptr[i] * b_ptr[i];
  }
}

template <>
void mul_backward_impl<Host>(Tensor& grad_output, Tensor& grad_input_1,
                             Tensor& grad_input_2, Tensor& a, Tensor& b) {
  auto grad_output_ptr = grad_output.data_ptr();
  auto grad_input_1_ptr = grad_input_1.data_ptr();
  auto grad_input_2_ptr = grad_input_2.data_ptr();
  auto a_ptr = a.data_ptr();
  auto b_ptr = b.data_ptr();

  for (size_t i = 0; i < a.numel(); i++) {
    // y = a * b, y'(a) = b * grad, y'(b) = a * grad
    grad_input_1_ptr[i] = b_ptr[i] * grad_output_ptr[i];
    grad_input_2_ptr[i] = a_ptr[i] * grad_output_ptr[i];
  }
}

}  // namespace tinytorch
