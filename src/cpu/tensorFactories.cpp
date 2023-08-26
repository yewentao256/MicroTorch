
#include "tensorFactories.hpp"

#include "tensor.hpp"
namespace microtorch {

// Tensor Create operators
Tensor zeros(std::vector<size_t> size, const std::string& device,
             bool requires_grad) {
  Tensor t(size, device, requires_grad);
  fill_scalar(t, 0);
  return t;
}

Tensor ones(std::vector<size_t> size, const std::string& device,
            bool requires_grad) {
  Tensor t(size, device, requires_grad);
  fill_scalar(t, 1);
  return t;
}

Tensor rand(std::vector<size_t> size, const std::string& device,
            bool requires_grad) {
  Tensor t(size, Device("cpu"), requires_grad);
  static std::mt19937 mersenne_engine{572547235};
  std::uniform_real_distribution<data_t> dist{0.f, 1.f};

  data_t* data_ptr = t.data_ptr();
  for (size_t i = 0; i < t.numel(); i++) {
    data_ptr[i] = dist(mersenne_engine);
  }
  if (device == "cuda") {
    return t.cuda();  // TODO: rand for cuda ops
  }
  return t;
}

template <>
void fill_impl<Host>(Tensor& self, const data_t value) {
  auto self_ptr = self.data_ptr();
  for (size_t i = 0; i < self.numel(); i++) {
    self_ptr[i] = value;
  }
}

template <>
void clone_impl<Host>(const Tensor& a, Tensor& out) {
  auto out_ptr = out.data_ptr();
  auto a_ptr = a.data_ptr();
  for (size_t i = 0; i < a.numel(); i++) {
    out_ptr[i] = a_ptr[i];
  }
}

template <>
void clone_backward_impl<Host>(const Tensor& grad_output, Tensor& grad_input) {
  auto grad_output_ptr = grad_output.data_ptr();
  auto grad_input_ptr = grad_input.data_ptr();
  for (size_t i = 0; i < grad_input.numel(); i++) {
    // y = a, y'(a) = 1 * grad
    grad_input_ptr[i] = grad_output_ptr[i];
  }
}

}  // namespace microtorch