/**
 * Copyright (c) 2022-2023 yewentao
 * Licensed under the MIT License.
 */
#include "cuda.hpp"
#include "functors.hpp"
#include "loops.cuh"
#include "ops.hpp"

namespace microtorch {
template <>
void neg_impl<Cuda>(TensorIterator& iter) {
  gpu_kernel(iter, binaryFunctor::Neg());
}
}  // namespace microtorch
