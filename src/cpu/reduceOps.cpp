/**
 * Copyright (c) 2022-2023 yewentao
 * Licensed under the MIT License.
 */
#include "reduceOps.hpp"

namespace microtorch {

template <>
void sum_impl<Host>(const Tensor& a, Tensor& out) {
  auto out_ptr = out.data_ptr();
  auto a_ptr = a.data_ptr();
  for (int64_t i = 0; i < a.numel(); i++) {
    out_ptr[0] += a_ptr[i];
  }
}

}  // namespace microtorch