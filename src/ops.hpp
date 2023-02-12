#pragma once

#include "tensor.hpp"
#include "context.hpp"

namespace tinytorch {

// tensor generation
Tensor zeros(size_t size, std::string device = "host");
Tensor ones(size_t size, std::string device = "host");
Tensor rand(size_t size, std::string device = "host");

// operators
Tensor operator+(Tensor& a, Tensor& b);
Tensor operator*(Tensor& a, Tensor& b);
Tensor operator-(Tensor& a, Tensor& b);
std::string repr(Tensor t);
std::ostream& operator<<(std::ostream& stream, Tensor t);

// Internal implementation of forward/backward
// Should NOT be called by the user
template <typename Device>
void add_impl(Context& ctx, Tensor& a, Tensor& b, Tensor& out);
std::vector<Tensor> add_backward_impl(Tensor grad_output);

template <typename Device>
void sub_impl(Context& ctx, Tensor& a, Tensor& b, Tensor& out);
std::vector<Tensor> sub_backward_impl(Tensor grad_output);

template <typename Device>
void mult_impl(Context& ctx, Tensor& a, Tensor& b, Tensor& out);
std::vector<Tensor> mult_backward_impl(Tensor a, Tensor b, Tensor grad_output);

Tensor square_impl(Tensor& a);
std::vector<Tensor> square_backward_impl(Tensor a, Tensor grad_output);

Tensor sum_impl(Tensor& a);
std::vector<Tensor> sum_backward_impl(size_t input_size, Tensor grad_output);

// Basic Tensor Math operators
// These operators should be called by the user and support Auto-Diff
Tensor square(Tensor& a);
Tensor sum(Tensor& a);
}  // namespace tinytorch
