
#include "unaryOps.hpp"

namespace tinytorch {

template <>
void square_impl<Host>(Tensor& a, Tensor& out) {
  auto out_ptr = out.data_ptr();
  auto a_ptr = a.data_ptr();
  for (size_t i = 0; i < a.numel(); i++) {
    out_ptr[i] = a_ptr[i] * a_ptr[i];
  }
}

std::vector<Tensor> square_backward_impl(Tensor a, Tensor& grad_output) {
  Tensor grad_input(a.shape(), grad_output.device());
  for (size_t i = 0; i < a.numel(); i++) {
    // TODO: multi dimension
    // y = a^2, y'(a) = 2 * a * grad
    grad_input[i] = 2 * a[i] * grad_output[i];
  }
  return {grad_input};
}

}  // namespace tinytorch
