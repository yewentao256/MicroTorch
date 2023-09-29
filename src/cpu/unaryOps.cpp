
#include "unaryOps.hpp"

namespace microtorch {

template <>
void square_impl<Host>(Tensor &a, Tensor &out) {
  auto out_ptr = out.data_ptr();
  auto a_ptr = a.data_ptr();
  for (int64_t i = 0; i < a.numel(); i++) {
    out_ptr[i] = a_ptr[i] * a_ptr[i];
  }
}

template <>
void square_backward_impl<Host>(Tensor &grad_output, Tensor &grad_input,
                                Tensor &input) {
  auto grad_output_ptr = grad_output.data_ptr();
  auto grad_input_ptr = grad_input.data_ptr();
  auto input_ptr = input.data_ptr();
  for (int64_t i = 0; i < input.numel(); i++) {
    // y = a^2, y'(a) = 2 * a * grad
    grad_input_ptr[i] = 2 * input_ptr[i] * grad_output_ptr[i];
  }
}

}  // namespace microtorch
