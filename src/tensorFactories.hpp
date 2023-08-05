#pragma once

#include "context.hpp"
#include "tensor.hpp"
#include "engine.hpp"

namespace tinytorch {

Tensor zeros(size_t size, const std::string& device);
Tensor zeros(std::vector<size_t> size, const std::string& device);
Tensor ones(size_t size, const std::string& device);
Tensor ones(std::vector<size_t> size, const std::string& device);
Tensor rand(size_t size, const std::string& device);
Tensor rand(std::vector<size_t> size, const std::string& device);

// Internal implementation
// Should NOT be called by the user
template <typename Device>
void fill_impl(Tensor& self, const data_t value);

// TODO: adding this to op repository? or we have to move this before `fill_`
template <>
void fill_impl<Host>(Tensor& self, const data_t value);

inline void fill_(Tensor& self, const data_t value){
  DISPATCH_OP(fill_impl, self.device(), self, value);
}

}  // namespace tinytorch
