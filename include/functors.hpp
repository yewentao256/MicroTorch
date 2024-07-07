/**
 * Copyright (c) 2022-2024 yewentao256
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

struct Sub {
  HOST_DEVICE INLINE data_t operator()(const data_t a, const data_t b) const {
    return a - b;
  }
};

struct Mul {
  HOST_DEVICE INLINE data_t operator()(const data_t a, const data_t b) const {
    return a * b;
  }
};

struct Div {
  HOST_DEVICE INLINE data_t operator()(const data_t a, const data_t b) const {
    return a / b;
  }
};

struct Neg {
  HOST_DEVICE INLINE data_t operator()(const data_t a) const {
    return -a;
  }
};

struct Clone {
  HOST_DEVICE INLINE data_t operator()(const data_t a) const {
    return a;
  }
};

}  // namespace ufunc

}  // namespace microtorch