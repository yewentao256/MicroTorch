/**
 * Copyright (c) 2022-2023 yewentao
 * Licensed under the MIT License.
 */
#pragma once

#include "macros.hpp"

namespace microtorch {

namespace binaryFunctor {

struct Add {
  HOST_DEVICE INLINE data_t operator()(const data_t a, const data_t b) const {
    return a + b;
  }
};
}  // namespace ufunc

}  // namespace microtorch