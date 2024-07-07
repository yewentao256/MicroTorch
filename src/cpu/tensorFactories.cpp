/**
 * Copyright (c) 2022-2024 yewentao256
 * Licensed under the MIT License.
 */
#include "tensorFactories.hpp"

#include "functors.hpp"
#include "loops.hpp"
#include "tensor.hpp"
#include "tensorIterator.hpp"

namespace microtorch {

template <>
void fill_impl<Host>(Tensor& self, const data_t value) {
  auto self_ptr = self.data_ptr();
  for (int64_t i = 0; i < self.numel(); i++) {
    self_ptr[i] = value;
  }
}

template <>
void clone_impl<Host>(TensorIterator& iter) {
  cpu_kernel(iter, binaryFunctor::Clone());
}

template <>
void rand_impl<Host>(Tensor& self) {
  static std::mt19937 mersenne_engine{572547235};
  std::uniform_real_distribution<data_t> dist{0.f, 1.f};

  data_t* data_ptr = self.data_ptr();
  for (int64_t i = 0; i < self.numel(); i++) {
    data_ptr[i] = dist(mersenne_engine);
  }
}

}  // namespace microtorch
