/**
 * Copyright (c) 2022-2024 yewentao256
 * Licensed under the MIT License.
 */
#pragma once

#include <bit>
#include <cstdint>
#include <type_traits>
#include <variant>
#include <vector>

#include "exception.hpp"
#include "tensor.hpp"
#include "array.hpp"

namespace microtorch {

/* An arg that supports all kinds of data type */
struct ArgT {
  using Variant = std::variant<data_t, Tensor, IntArrayRef, bool>;
  Variant value;

  ArgT(Variant v) : value(v) {}

  template <typename T, typename... Args>
  static void fill(std::vector<ArgT>& inputs, T&& arg, Args&&... args) {
    inputs.emplace_back(std::forward<T>(arg));
    fill(inputs, std::forward<Args>(args)...);
  }

  static void fill(std::vector<ArgT>& inputs) {}
};

template <typename... Args>
std::vector<ArgT> make_arg_list(Args&&... args) {
  std::vector<ArgT> arg_list;
  ArgT::fill(arg_list, std::forward<Args>(args)...);
  return arg_list;
}

inline IntArrayRef calculate_init_stride(const IntArrayRef& shape) {
  IntArrayRef strides(shape.size());
  int64_t stride = 1;
  for (int i = shape.size() - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

}  // namespace microtorch
