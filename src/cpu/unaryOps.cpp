/**
 * Copyright (c) 2022-2023 yewentao
 * Licensed under the MIT License.
 */
#include "unaryOps.hpp"

#include "loops.hpp"
#include "tensorIterator.hpp"
#include "functors.hpp"

namespace microtorch {

template <>
void neg_impl<Host>(TensorIterator& iter) {
  cpu_kernel(iter, binaryFunctor::Neg());
}

}  // namespace microtorch
