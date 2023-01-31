#pragma once

#include "tensor.hpp"
namespace tinytorch {
// tensor generation
Tensor zero(size_t size);
Tensor ones(size_t size);
Tensor rand(size_t size);
std::string repr(Tensor t);
std::ostream& operator<<(std::ostream& stream, Tensor t);

// operators+
Tensor operator+(Tensor a, Tensor b);
Tensor operator*(Tensor a, Tensor b);
Tensor operator-(Tensor a, Tensor b);

// Internal implementation of forward/backward
// Should NOT be called by the user
Tensor add_impl(Tensor a, Tensor b);
std::vector<Tensor> add_backward_impl(Tensor grad_output);

Tensor sub_impl(Tensor a, Tensor b);
std::vector<Tensor> sub_backward_impl(Tensor grad_output);

Tensor mult_impl(Tensor a, Tensor b);
std::vector<Tensor> mult_backward_impl(Tensor a, Tensor b, Tensor grad_output);

Tensor square_impl(Tensor a);
std::vector<Tensor> square_backward_impl(Tensor a, Tensor grad_output);

Tensor sum_impl(Tensor a);
std::vector<Tensor> sum_backward_impl(size_t input_size, Tensor grad_output);

// Basic Tensor Math operators
// These operators should be called by the user and support Auto-Diff
Tensor square(Tensor a);
Tensor sum(Tensor a);
}  // namespace tinytorch
