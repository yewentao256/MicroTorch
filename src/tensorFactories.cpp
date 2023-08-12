
#include "tensorFactories.hpp"
#include "tensor.hpp"
namespace tinytorch {

// Tensor Create operators
Tensor zeros(size_t size, const std::string& device) {
  return zeros(std::vector<size_t>{size}, device);
}

Tensor zeros(std::vector<size_t> size, const std::string& device) {
  Tensor t(size, device);
  fill_scalar(t, 0);
  return t;
}

Tensor ones(size_t size, const std::string& device) {
  return ones(std::vector<size_t>{size}, device);
}

Tensor ones(std::vector<size_t> size, const std::string& device) {
  Tensor t(size, device);
  fill_scalar(t, 1);
  return t;
}

Tensor rand(size_t size, const std::string& device) {
  return rand(std::vector<size_t>{size}, device);
}

Tensor rand(std::vector<size_t> size, const std::string& device) {
  Tensor t(size);
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

}  // namespace tinytorch
