
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
void add_backward_impl<Host>(Tensor& grad_output, Tensor& dx_1,
                             Tensor& dx_2) {
  for (size_t i = 0; i < grad_output.numel(); i++) {
    // TODO: multi shape
    // y = a + b, y'(a) = 1 * grad
    dx_1[i] = grad_output[i];
    dx_2[i] = grad_output[i];
  }
}

Tensor Tensor::operator+(const Tensor& other) {
  Tensor out = zeros(this->shape(), this->device());
  add_out(*this, other, out);
  return out;
}

Tensor& Tensor::operator+=(const Tensor& other) {
  add_out(*this, other, *this);
  return *this;
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

std::vector<Tensor> sub_backward_impl(Tensor& grad_output) {
  Tensor grad_input_a(grad_output.shape(), grad_output.device());
  Tensor grad_input_b(grad_output.shape(), grad_output.device());
  // TODO: multi shape
  for (size_t i = 0; i < grad_output.numel(); i++) {
    // y = a - b, y'(a) = 1 * grad, y'(b) = -1 * grad
    grad_input_a[i] = grad_output[i];
    grad_input_b[i] = -grad_output[i];
  }
  return {grad_input_a, grad_input_b};
}

Tensor Tensor::operator-(const Tensor& other) {
  // TODO：注意这里operator-返回一个新的Tensor对象，而不是引用。这可以通过返回值优化（Return
  // Value Optimization，RVO）或命名返回值优化（Named Return Value
  // Optimization，NRVO）来高效地完成
  // 对于Tensor这样的类型，通常会禁用复制构造函数，因为复制一个Tensor可能会涉及到大量数据的复制，这可能会非常耗时。相反，Tensor通常会提供移动构造函数。这也是可以被RVO来优化的
  // 需要检查一下目前tensor的复制构造和移动构造
  Tensor out = zeros(this->shape(), this->device());
  sub_out(*this, other, out);
  return out;
}

Tensor& Tensor::operator-=(const Tensor& other) {
  sub_out(*this, other, *this);
  return *this;
}

template <>
void mult_impl<Host>(Tensor& a, Tensor& b, Tensor& out) {
  auto out_ptr = out.data_ptr();
  auto a_ptr = a.data_ptr();
  auto b_ptr = b.data_ptr();
  for (size_t i = 0; i < a.numel(); i++) {
    out_ptr[i] = a_ptr[i] * b_ptr[i];
  }
}

std::vector<Tensor> mult_backward_impl(Tensor a, Tensor b,
                                       Tensor& grad_output) {
  Tensor grad_input_a(grad_output.shape(), grad_output.device());
  Tensor grad_input_b(grad_output.shape(), grad_output.device());
  for (size_t i = 0; i < a.numel(); i++) {
    // TODO: multi shape
    // y = a * b, y'(a) = b * grad
    grad_input_a[i] = b[i] * grad_output[i];
    grad_input_b[i] = a[i] * grad_output[i];
  }
  return {grad_input_a, grad_input_b};
}

Tensor Tensor::operator*(const Tensor& other) {
  Tensor out = zeros(this->shape(), this->device());
  mul_out(*this, other, out);
  return out;
}

Tensor& Tensor::operator*=(const Tensor& other) {
  mul_out(*this, other, *this);
  return *this;
}

}  // namespace tinytorch
