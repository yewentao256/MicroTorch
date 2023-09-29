#pragma once

#include <variant>
#include <vector>

#include "exception.hpp"
#include "tensor.hpp"

namespace microtorch {

/* An arg that supports all kinds of data type */
struct ArgT {
  using Variant = std::variant<data_t, Tensor>;
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

inline void check_device_shape(const std::vector<Tensor>& inputs) {
  int64_t len_inputs = inputs.size();
  Device device = inputs[0].device();
  auto shape = inputs[0].shape();
  for (int64_t i = 1; i < len_inputs; i++) {
    TORCH_CHECK(inputs[i].device() == device,
                "all the tensors should be in the same device.");
    TORCH_CHECK(inputs[i].shape() == shape,
                "size of the tensors should be the same");
    // TODO: support broadcast through a general iterator
    // And that iterator can also compute device and shape (broadcast).
  }
}

}  // namespace microtorch
