#pragma once

#include "ops.hpp"

namespace tinytorch {

// Internal implementation of forward/backward
// Should NOT be called by the user
Tensor add_cuda_impl(Tensor a, Tensor b);
std::vector<Tensor> add_backward_cuda_impl(Tensor grad_output);

/* Tensor sub_impl(Tensor a, Tensor b);
std::vector<Tensor> sub_backward_impl(Tensor grad_output);

Tensor mult_impl(Tensor a, Tensor b);
std::vector<Tensor> mult_backward_impl(Tensor a, Tensor b, Tensor grad_output);

Tensor square_impl(Tensor a);
std::vector<Tensor> square_backward_impl(Tensor a, Tensor grad_output);

Tensor sum_impl(Tensor a);
std::vector<Tensor> sum_backward_impl(int input_size, Tensor grad_output);

// Basic Tensor Math operators
// These operators should be called by the user and support Auto-Diff
Tensor square(Tensor a);
Tensor sum(Tensor a); */
}  // namespace tinytorch
