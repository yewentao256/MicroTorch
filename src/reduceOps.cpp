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

}  // namespace tinytorch