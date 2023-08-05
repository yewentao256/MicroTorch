#include "reduceOps.hpp"

namespace tinytorch {

template <>
void sum_impl<Host>(Tensor& a, Tensor& out) {
  auto out_ptr = out.data_ptr();
  auto a_ptr = a.data_ptr();
  for (size_t i = 0; i < a.numel(); i++) {
    out_ptr[0] += a_ptr[i];
  }
}
std::vector<Tensor> sum_backward_impl(Tensor& input, Tensor& grad_output) {
  TORCH_CHECK(grad_output.numel() == 1, "grad_output size should equal to 1");
  Tensor grad_input(input.shape());
  fill_(grad_input, grad_output[0]);
  return {grad_input};
}

}  // namespace tinytorch