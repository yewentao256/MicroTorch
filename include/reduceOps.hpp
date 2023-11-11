/**
 * Copyright (c) 2022-2023 yewentao
 * Licensed under the MIT License.
 */
#pragma once

#include "context.hpp"
#include "engine.hpp"
#include "graph.hpp"
#include "tensor.hpp"
#include "tensorFactories.hpp"
#include "tensorIterator.hpp"

constexpr size_t bitset_size = 64;

namespace microtorch {

template <typename Device>
void sum_impl(const Tensor& a, Tensor& out);
template <typename Device>
void sum_dim_impl(const Tensor& a, Tensor& out, IntArrayRef& dims,
                  bool keep_dim = false);

Tensor sum(const Tensor& a);
Tensor sum_dim(const Tensor& a, IntArrayRef dims, bool keep_dim = false);

inline std::bitset<bitset_size> make_dim_mask(IntArrayRef& dims, int64_t ndim) {
  std::bitset<bitset_size> mask;
  if (dims.empty()) {
    mask = std::bitset<bitset_size>().flip();
  } else {
    TORCH_CHECK(ndim <= bitset_size, "only tensors with up to ", bitset_size,
                " dims are supported");
    for (const auto i : irange(dims.size())) {
      size_t dim = dims[i];
      TORCH_CHECK(!mask[dim], "dim ", dim,
                  " appears multiple times in the list of dims");
      mask[dim] = true;
    }
  }
  return mask;
}

// Infer the actual result Tensor for reduction, new storage is required.
inline Tensor infer_reduce_tensor(const Tensor& self,
                                  std::bitset<bitset_size> mask, bool keepdim) {
  std::vector<int64_t> shape = self.shape().vec();
  for (int dim = shape.size() - 1; dim >= 0; dim--) {
    if (mask[dim]) {
      if (keepdim) {
        shape[dim] = 1;
      } else {
        shape.erase(shape.begin() + dim);
      }
    }
  }
  return zeros(shape, self.device(), self.requires_grad());
}

// Build a view tensor for TensorIterator
inline Tensor view_reduce_result(const Tensor& result, int ndim,
                                 std::bitset<bitset_size> mask, bool keepdim) {
  if (keepdim) {
    return result;
  }
  std::vector<int64_t> shape = result.shape().vec();
  std::vector<int64_t> stride = result.stride().vec();
  for (const auto dim : irange(ndim)) {
    if (mask[dim]) {
      shape.insert(shape.begin() + dim, 1);
      stride.insert(stride.begin() + dim, 0);
    }
  }
  return result.as_strided(shape, stride);
}

}  // namespace microtorch
