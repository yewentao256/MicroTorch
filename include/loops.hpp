/**
 * Copyright (c) 2022-2024 yewentao256
 * Licensed under the MIT License.
 */

#pragma once

#include <stdint.h>

#include <cstring>
#include <utility>

#include "exception.hpp"
#include "funcTraits.hpp"
#include "irange.hpp"
#include "load.hpp"
#include "tensorIterator.hpp"

namespace microtorch {

namespace internal {
#ifdef USE_CUDA

template <class F, class Tuple, std::size_t... INDEX>
constexpr decltype(auto) apply_impl(F&& f, Tuple&& t,
                                    std::index_sequence<INDEX...>) {
  return std::forward<F>(f)(std::get<INDEX>(std::forward<Tuple>(t))...);
}

template <class F, class Tuple>
constexpr decltype(auto) apply(F&& f, Tuple&& t) {
  return apply_impl(
      std::forward<F>(f), std::forward<Tuple>(t),
      std::make_index_sequence<
          std::tuple_size<std::remove_reference_t<Tuple>>::value>{});
}
#else

template <class F, class Tuple>
inline constexpr decltype(auto) apply(F&& f, Tuple&& t) {
  return std::apply(std::forward<F>(f), std::forward<Tuple>(t));
}

#endif
}  // namespace internal

template <typename traits, std::size_t... INDEX>
typename traits::ArgsTuple dereference_impl(char* data[],
                                            const int64_t* strides, int64_t i,
                                            std::index_sequence<INDEX...>) {
  return std::make_tuple(load<typename traits::template arg<INDEX>::type>(
      data[INDEX] + i * strides[INDEX])...);
}

template <typename traits>
typename traits::ArgsTuple dereference(char* data[], const int64_t* strides,
                                       int64_t i) {
  using Indices = std::make_index_sequence<traits::num_args>;
  return dereference_impl<traits>(data, strides, i, Indices{});
}

// SFINAE: Valid when result_type of func_t is not `void`
template <
    typename func_t,
    std::enable_if_t<!std::is_void_v<typename FuncTraits<func_t>::result_type>,
                     int> = 0>
static inline void execute_op(char* data[], const int64_t* strides, int64_t i,
                              int64_t n, func_t&& op) {
  using traits = FuncTraits<func_t>;
  using result_type = typename traits::result_type;
  for (; i < n; i++) {
    result_type* out_ptr = (result_type*)(data[0] + i * strides[0]);
    *out_ptr = internal::apply(std::forward<func_t>(op),
                               dereference<traits>(&data[1], &strides[1], i));
  }
}

// SFINAE: Valid when result_type of func_t is `void`
template <
    typename func_t,
    std::enable_if_t<std::is_void_v<typename FuncTraits<func_t>::result_type>,
                     int> = 0>
static inline void execute_op(char* data[], const int64_t* strides, int64_t i,
                              int64_t n, func_t&& op) {
  using traits = FuncTraits<func_t>;
  for (; i < n; i++) {
    internal::apply(std::forward<func_t>(op),
                    dereference<traits>(&data[0], &strides[0], i));
  }
}

// Basic loop operation (one output, N inputs). May be auto-vectorized
// by the compiler. Supports inputs and outputs of different types.
template <typename func_t>
static inline void basic_loop(char* data[], const int64_t* strides_, int64_t i,
                              int64_t n, func_t&& op) {
  using traits = FuncTraits<func_t>;
  constexpr int ntensors = traits::num_args + 1;

  // Copying strides to temporary array helps auto vectorization in older GCC
  // versions.
  int64_t strides[ntensors];
  for (const auto i : irange(ntensors)) {
    strides[i] = strides_[i];
  }

  execute_op(data, strides, i, n, std::forward<func_t>(op));
}

template <typename func_t>
void cpu_kernel(TensorIterator& iter, func_t&& op,
                int64_t grain_size = GRAIN_SIZE) {
  using traits = FuncTraits<func_t>;
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::num_args);
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);

  iter.for_each(
      [&](char** data, const int64_t* strides, int64_t n) {
        // basic loop can handle 1d slices with arbitrary strides, and 1d slices
        // is all that iter.for_each is ever sending to the loop lambda
        basic_loop(data, strides, 0, n, std::forward<func_t>(op));
      },
      grain_size);
}

template <typename func_t>
void cpu_serial_kernel(TensorIterator& iter, func_t&& op, const Range& range) {
  using traits = FuncTraits<func_t>;
  constexpr bool result_void = std::is_void_v<typename traits::result_type>;
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::num_args &&
                        ((result_void && iter.noutputs() == 0) ||
                         (!result_void && iter.noutputs() == 1)));

  iter.serial_for_each(
      [&](char** data, const int64_t* strides, int64_t n) {
        basic_loop(data, strides, 0, n, std::forward<func_t>(op));
      },
      range);
}

template <typename func_t>
void cpu_serial_kernel(TensorIterator& iter, func_t&& op) {
  cpu_serial_kernel(iter, op, {0, iter.numel()});
}

}  // namespace microtorch
