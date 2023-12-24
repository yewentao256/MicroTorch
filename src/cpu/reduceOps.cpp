/**
 * Copyright (c) 2022-2023 yewentao
 * Licensed under the MIT License.
 */
#include "reduceOps.hpp"

#include <bitset>

#include "accumulateType.hpp"
#include "tensorIterator.hpp"
#include "utils.hpp"
#include "vectorized.hpp"

namespace microtorch {

template <typename T>
T CeilLog2(const T &x) {
  TORCH_INTERNAL_ASSERT(std::is_integral_v<T>);
  if (x <= 2) {
    return 1;
  }
  // std::bit_width returns the highest bit location + 1
  return static_cast<T>(std::bit_width(static_cast<uint64_t>(x) - 1));
}

template <typename scalar_t>
struct LoadPolicy {
  static constexpr int64_t memsize() { return sizeof(scalar_t); }

  static scalar_t load(const char *data, int64_t stride, int64_t index) {
    auto *ptr = reinterpret_cast<const scalar_t *>(data + index * stride);
    return *ptr;
  }
};

template <typename scalar_t>
struct LoadPolicy<Vectorized<scalar_t>> {
  static constexpr int64_t memsize() {
    return sizeof(scalar_t) * Vectorized<scalar_t>::size();
  }

  static Vectorized<scalar_t> load(const char *data, int64_t stride,
                                   int64_t index) {
    auto *ptr = data + index * stride;
    return Vectorized<scalar_t>::loadu(ptr);
  }
};

template <typename acc_t>
struct CastLoadPolicy {
  static constexpr int64_t memsize() { return sizeof(data_t); }

  static acc_t load(const char *data, int64_t stride, int64_t index) {
    const auto val = LoadPolicy<data_t>::load(data, stride, index);
    return acc_t(val);
  }
};

// For inner sum, load full vec_t then sum partials down to vacc_t size
template <typename vec_t, typename vacc_t>
struct InnerSumCastLoadPolicy {
  using scalar_t = vechold_type<vec_t>;
  using acc_t = vechold_type<vacc_t>;

  static constexpr int64_t memsize() { return LoadPolicy<vec_t>::memsize(); }

  static vacc_t load(const char *data, int64_t stride, int64_t index) {
    auto ptr = reinterpret_cast<const scalar_t *>(data + stride * index);
    return load_reduce_vec<acc_t>(
        ptr, [](acc_t a, scalar_t b) { return a + b; }, acc_t(0));
  }
};

// For outer sum, load a partial vec_t of size vacc_t then cast to vacc_t
template <typename vec_t, typename vacc_t>
struct OuterSumCastLoadPolicy {
  using scalar_t = vechold_type<vec_t>;
  using acc_t = vechold_type<vacc_t>;

  static constexpr int64_t memsize() {
    return sizeof(scalar_t) * vacc_t::size();
  }

  static vacc_t load(const char *data, int64_t stride, int64_t index) {
    static_assert(vacc_t::size() <= vec_t::size(), "");
    const auto val = vec_t::loadu(data + stride * index, vacc_t::size());
    alignas(64) scalar_t values[vec_t::size()];
    val.store(values);

    alignas(64) acc_t acc[vacc_t::size()];
    for (const auto i : irange(vacc_t::size())) {
      acc[i] = values[i];
    }

    return vacc_t::loadu(acc);
  }
};

template <typename acc_t>
struct CastStoreAccumulate {
  static void store(char *data, int64_t stride, int64_t index, acc_t value) {
    auto *ptr = reinterpret_cast<data_t *>(data + index * stride);
    *ptr += value;
  }
};

template <typename StorePolicy, typename scalar_t>
static void store(char *data, int64_t stride, int64_t index, scalar_t value) {
  StorePolicy::store(data, stride, index, value);
}

template <typename StorePolicy, typename scalar_t, size_t numel>
static void store(char *data, int64_t stride, int64_t index,
                  const std::array<scalar_t, numel> &values) {
  auto *base_ptr = data + stride * index;
  for (const auto k : irange(numel)) {
    auto val = values[k];
    StorePolicy::store(base_ptr, stride, k, val);
  }
}

template <typename StorePolicy, typename scalar_t>
static void store(char *data, int64_t stride, int64_t index,
                  const Vectorized<scalar_t> &values) {
  using vec_t = Vectorized<scalar_t>;
  alignas(64) std::array<scalar_t, vec_t::size()> array_values;
  values.store(array_values.data());
  store<StorePolicy>(data, stride, index, array_values);
}

/** Simultaneously sum over n rows at once

This algorithm calculates the sum without loss of precision over large axes. It
does this by chunking the sum into groups of 16 or more elements. The sums of
these chunks are also summed in chunks and so on until there is just a single
sum value remaining. This means only numbers of a similar order of magnitude are
added together, thus minimising rounding errors.

This is done in a single linear pass over the data and with O(1) extra storage.
A simplified recursive implementation would look like this:

  scalar_t row_sum(const scalar_t * data, int64_t n) {
    // Note, in practice the chunk size can increase with n
    // This allows the recursion depth to be limited to O(1).
    constexpr int64_t min_chunk_size = 16;

    scalar_t sum = 0;
    if (n <= min_chunk_size) {
      // Recursive base case, calculate a simple running sum
      for (const auto i : irange(n)) {
        sum += data[i];
      }
      return sum;
    }

    // Recursively sum larger chunks of elements
    const int64_t chunk_size = std::max(divup(n, min_chunk_size),
min_chunk_size); for (int64_t i = 0; i < n; i += chunk_size) { sum +=
row_sum(data + i, std::min(chunk_size, n - i));
    }
    return sum;
  }
*/
template <typename scalar_t, int64_t nrows, typename LoadPolicy>
std::array<scalar_t, nrows> multi_row_sum(const char *in_data,
                                          const int64_t row_stride,
                                          const int64_t col_stride,
                                          const int64_t size) {
  constexpr int64_t num_levels = 4;

  const int64_t level_power = std::max(int64_t(4), CeilLog2(size) / num_levels);
  const int64_t level_step = (1 << level_power);
  const int64_t level_mask = level_step - 1;

  scalar_t acc[num_levels][nrows];
  std::fill_n(&acc[0][0], num_levels * nrows, scalar_t(0));

  int64_t i = 0;
  for (; i + level_step <= size;) {
    for (int64_t j = 0; j < level_step; ++j, ++i) {
      const char *sum_base = in_data + i * row_stride;
#if !defined(COMPILING_FOR_MIN_SIZE)
#pragma unroll
#endif
      for (const auto k : irange(nrows)) {
        acc[0][k] += LoadPolicy::load(sum_base, col_stride, k);
      }
    }

    for (const auto j : irange(1, num_levels)) {
#if !defined(COMPILING_FOR_MIN_SIZE)
#pragma unroll
#endif
      for (const auto k : irange(nrows)) {
        acc[j][k] += acc[j - 1][k];
        acc[j - 1][k] = scalar_t(0);
      }

      const auto mask = (level_mask << (j * level_power));
      if ((i & mask) != 0) {
        break;
      }
    }
  }

  for (; i < size; ++i) {
    const char *sum_base = in_data + i * row_stride;
#if !defined(COMPILING_FOR_MIN_SIZE)
#pragma unroll
#endif
    for (const auto k : irange(nrows)) {
      acc[0][k] += LoadPolicy::load(sum_base, col_stride, k);
    }
  }

  for (const auto j : irange(1, num_levels)) {
#if !defined(COMPILING_FOR_MIN_SIZE)
#pragma unroll
#endif
    for (const auto k : irange(nrows)) {
      acc[0][k] += acc[j][k];
    }
  }

  std::array<scalar_t, nrows> ret;
  for (const auto k : irange(nrows)) {
    ret[k] = acc[0][k];
  }
  return ret;
}

template <typename scalar_t, typename LoadPolicy>
scalar_t row_sum(const char *in_data, const int64_t in_stride,
                 const int64_t size) {
  constexpr int64_t ilp_factor = 4;

  // Interpret row as a (-1, ilp_factor) shaped array to find partial sums
  const int64_t size_ilp = size / ilp_factor;
  auto partial_sums = multi_row_sum<scalar_t, ilp_factor, LoadPolicy>(
      in_data, in_stride * ilp_factor, in_stride, size_ilp);

  for (int64_t i = size_ilp * ilp_factor; i < size; ++i) {
    partial_sums[0] += LoadPolicy::load(in_data, in_stride, i);
  }

  for (const auto k : irange(1, ilp_factor)) {
    partial_sums[0] += partial_sums[k];
  }

  return partial_sums[0];
}

template <typename acc_t, typename VecLoadPolicy, typename ScalarLoadPolicy,
          typename StorePolicy>
void vectorized_inner_sum(char *data[2], int64_t outer_stride,
                          int64_t out_stride, int64_t size0, int64_t size1) {
  using vacc_t = Vectorized<acc_t>;
  constexpr int64_t vec_stride = VecLoadPolicy::memsize();
  constexpr int64_t scalar_stride = ScalarLoadPolicy::memsize();
  constexpr int64_t vec_numel = vec_stride / scalar_stride;
  const int64_t vec_size = size0 / vec_numel;

  // Input is contiguous over the first (reduced) dimension
  for (const auto j : irange(size1)) {
    const auto *row_in = data[1] + j * outer_stride;
    auto vec_acc = row_sum<vacc_t, VecLoadPolicy>(row_in, vec_stride, vec_size);

    acc_t final_acc = 0;
    for (int64_t k = vec_size * vec_numel; k < size0; ++k) {
      final_acc += ScalarLoadPolicy::load(row_in, scalar_stride, k);
    }

    alignas(64) std::array<acc_t, vacc_t::size()> partials{};
    vec_acc.store(partials.data());
    for (const auto k : irange(partials.size())) {
      final_acc += partials[k];
    }
    store<StorePolicy>(data[0], out_stride, j, final_acc);
  }
}

template <typename acc_t, typename LoadPolicy, typename StorePolicy>
void scalar_inner_sum(char *data[2], int64_t in_strides[2], int64_t out_stride,
                      int64_t size0, int64_t size1) {
  for (const auto j : irange(size1)) {
    const auto *row_in = data[1] + j * in_strides[1];
    auto ans = row_sum<acc_t, LoadPolicy>(row_in, in_strides[0], size0);
    store<StorePolicy>(data[0], out_stride, j, ans);
  }
}

template <typename acc_t, typename VecLoadPolicy, typename ScalarLoadPolicy,
          typename StorePolicy>
void vectorized_outer_sum(char *data[2], int64_t inner_stride,
                          int64_t out_stride, int64_t size0, int64_t size1) {
  using vacc_t = Vectorized<acc_t>;
  constexpr int64_t scalar_stride = ScalarLoadPolicy::memsize();
  constexpr int64_t vec_stride = VecLoadPolicy::memsize();
  constexpr int64_t nrows = 4;

  // Input is contiguous over the second (non-reduced) dimension
  int64_t j = 0;
  for (; j + nrows * vacc_t::size() <= size1; j += nrows * vacc_t::size()) {
    const auto *row_in = data[1] + j * scalar_stride;
    auto sums = multi_row_sum<vacc_t, nrows, VecLoadPolicy>(
        row_in, inner_stride, vec_stride, size0);

    for (const auto i : irange(nrows)) {
      const int64_t base_idx = j + i * vacc_t::size();
      store<StorePolicy>(data[0], out_stride, base_idx, sums[i]);
    }
  }

  for (; j + vacc_t::size() <= size1; j += vacc_t::size()) {
    const auto *row_in = data[1] + j * scalar_stride;
    const vacc_t sums =
        row_sum<vacc_t, VecLoadPolicy>(row_in, inner_stride, size0);

    store<StorePolicy>(data[0], out_stride, j, sums);
  }

  for (; j < size1; ++j) {
    const auto *row_in = data[1] + j * scalar_stride;
    auto ans = row_sum<acc_t, ScalarLoadPolicy>(row_in, inner_stride, size0);
    store<StorePolicy>(data[0], out_stride, j, ans);
  }
}

template <typename acc_t, typename LoadPolicy, typename StorePolicy>
void scalar_outer_sum(char *data[2], int64_t in_strides[2], int64_t out_stride,
                      int64_t size0, int64_t size1) {
  constexpr int64_t nrows = 4;
  int64_t j = 0;
  // batch handling sum, batch_size = nrows
  for (; j + (nrows - 1) < size1; j += nrows) {
    const auto *row_in = data[1] + j * in_strides[1];
    auto sums = multi_row_sum<acc_t, nrows, LoadPolicy>(row_in, in_strides[0],
                                                        in_strides[1], size0);
    store<StorePolicy>(data[0], out_stride, j, sums);
  }

  // handling the rest of sum
  for (; j < size1; ++j) {
    const auto *row_in = data[1] + j * in_strides[1];
    auto ans = row_sum<acc_t, LoadPolicy>(row_in, in_strides[0], size0);
    store<StorePolicy>(data[0], out_stride, j, ans);
  }
}

template <>
void sum_impl<Host>(const Tensor &a, Tensor &out) {
  auto out_ptr = out.data_ptr();
  auto a_ptr = a.data_ptr();
  for (int64_t i = 0; i < a.numel(); i++) {
    out_ptr[0] += a_ptr[i];
  }
}

template <typename F>
static inline void unary_outer_loop(char *data[2], const int64_t strides[2],
                                    int64_t n, F f) {
  for (const auto j : irange(n)) {
    (void)j;  // unused variable.
    f();
    data[0] += strides[0];
    data[1] += strides[1];
  }
}

void cascade_sum(TensorIterator &iter) {
  iter.parallel_reduce([&](char **data, const int64_t *strides, int64_t size0,
                           int64_t size1) {
    int64_t in_strides[] = {strides[1], strides[3]};
    int64_t out_strides[] = {strides[0], strides[2]};

    // Move reduction to be the 1st dim
    if (out_strides[0] != 0 && out_strides[1] == 0) {
      std::swap(in_strides[0], in_strides[1]);
      std::swap(out_strides[0], out_strides[1]);
      std::swap(size0, size1);
    }

    // Special case? - not a true reduction
    if (out_strides[0] != 0 && out_strides[1] != 0) {
      int64_t outer_strides[] = {strides[2], strides[3]};
      unary_outer_loop(data, outer_strides, size1, [&] {
        char *ptrs[3] = {data[0], data[0], data[1]};
        int64_t inner_strides[3] = {strides[0], strides[0], strides[1]};
        basic_loop(ptrs, inner_strides, 0, size0,
                   [](data_t a, data_t b) { return a + b; });
      });
      return;
    }

    const int64_t out_stride = out_strides[1];
    TORCH_INTERNAL_ASSERT(out_strides[0] == 0);

    using vec_t = Vectorized<data_t>;
    using acc_t = acc_type<data_t, true>;
    using vacc_t = Vectorized<acc_t>;
    using ScalarLoadPolicy = CastLoadPolicy<acc_t>;
    using StorePolicy = CastStoreAccumulate<acc_t>;

    if (in_strides[0] == sizeof(data_t) && size0 >= vec_t::size()) {
      // Contiguous inner reduction
      using VecLoadPolicy = InnerSumCastLoadPolicy<vec_t, vacc_t>;
      vectorized_inner_sum<acc_t, VecLoadPolicy, ScalarLoadPolicy, StorePolicy>(
          data, in_strides[1], out_stride, size0, size1);
    } else if (in_strides[1] == sizeof(data_t) && size1 >= vec_t::size()) {
      // Contiguous outer reduction
      using VecLoadPolicy = OuterSumCastLoadPolicy<vec_t, vacc_t>;
      vectorized_outer_sum<acc_t, VecLoadPolicy, ScalarLoadPolicy, StorePolicy>(
          data, in_strides[0], out_stride, size0, size1);
    } else if (in_strides[0] < in_strides[1]) {
      scalar_inner_sum<acc_t, ScalarLoadPolicy, StorePolicy>(
          data, in_strides, out_stride, size0, size1);
    } else {
      scalar_outer_sum<acc_t, ScalarLoadPolicy, StorePolicy>(
          data, in_strides, out_stride, size0, size1);
    }
  });
}

inline std::bitset<bitset_size> make_dim_mask(IntArrayRef &dims, int64_t ndim) {
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
inline Tensor infer_reduce_tensor(const Tensor &self,
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
inline Tensor view_reduce_result(const Tensor &result, int ndim,
                                 std::bitset<bitset_size> mask, bool keepdim) {
  if (keepdim) {
    return result;
  }
  std::vector<int64_t> shape = result.shape().vec();
  std::vector<int64_t> stride = result.stride().vec();
  for (const auto i : irange(ndim)) {
    if (mask[i]) {
      shape.insert(shape.begin() + i, 1);
      stride.insert(stride.begin() + i, 0);
    }
  }
  return result.as_strided(shape, stride);
}

template <>
void sum_dim_impl<Host>(const Tensor &a, Tensor &out, IntArrayRef &dims,
                        bool keep_dim) {
  int64_t ndim = a.ndim();
  auto mask = make_dim_mask(dims, ndim);
  auto result = infer_reduce_tensor(a, mask, keep_dim);
  auto viewed_result = view_reduce_result(result, ndim, mask, keep_dim);
  TensorIterator iter;
  iter.resize_outs(false).is_reduction(true);
  iter.add_output(viewed_result).add_input(a).build();
  cascade_sum(iter);
  out = result;  // TODO: temporily use this, fix this later
}

}  // namespace microtorch