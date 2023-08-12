#pragma once

#include "context.hpp"
#include "tensor.hpp"
#include "engine.hpp"

namespace tinytorch {

// Internal implementation
template <typename Device>
void fill_impl(Tensor& self, const data_t value);
template <>
void fill_impl<Host>(Tensor& self, const data_t value);
template <>
void fill_impl<Cuda>(Tensor& self, const data_t value);

// user interface
inline void fill_scalar(Tensor& self, const data_t value){
  DISPATCH_OP(fill_impl, self.device(), self, value);
}

Tensor zeros(size_t size, const std::string& device);
Tensor zeros(std::vector<size_t> size, const std::string& device);
Tensor ones(size_t size, const std::string& device);
Tensor ones(std::vector<size_t> size, const std::string& device);
Tensor rand(size_t size, const std::string& device);
Tensor rand(std::vector<size_t> size, const std::string& device);

}  // namespace tinytorch
