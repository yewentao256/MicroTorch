#pragma once

#include "tensor.hpp"
#include "context.hpp"

namespace tinytorch {

// tensor generation
Tensor zeros(size_t size, std::string device = "cpu");
Tensor ones(size_t size, std::string device = "cpu");
Tensor rand(size_t size, std::string device = "cpu");

// operators
Tensor operator+(Tensor& a, Tensor& b);
Tensor operator*(Tensor& a, Tensor& b);
Tensor operator-(Tensor& a, Tensor& b);
std::string repr(Tensor t, size_t print_size = 20, const std::string& name = "name");
std::ostream& operator<<(std::ostream& stream, Tensor t);

// Internal implementation of forward/backward
// Should NOT be called by the user
template <typename Device>
void add_impl(Context& ctx, Tensor& a, Tensor& b, Tensor& out);
template <typename Device>
void add_backward_impl(Context& ctx, Tensor& dy, Tensor& dx_1, Tensor& dx_2);

template <typename Device>
void sub_impl(Context& ctx, Tensor& a, Tensor& b, Tensor& out);
std::vector<Tensor> sub_backward_impl(Tensor& dy);

template <typename Device>
void mult_impl(Context& ctx, Tensor& a, Tensor& b, Tensor& out);
std::vector<Tensor> mult_backward_impl(Tensor a, Tensor b, Tensor& dy);

template <typename Device>
void square_impl(Context& ctx, Tensor& a, Tensor& out);
std::vector<Tensor> square_backward_impl(Tensor a, Tensor& dy);

template <typename Device>
void sum_impl(Context& ctx, Tensor& a, Tensor& out);
std::vector<Tensor> sum_backward_impl(size_t input_size, Tensor& dy);

// Basic Tensor Math operators
// These operators should be called by the user and support Auto-Diff
Tensor square(Tensor& a);
Tensor sum(Tensor& a);
}  // namespace tinytorch
