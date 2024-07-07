/**
 * Copyright (c) 2022-2024 yewentao256
 * Licensed under the MIT License.
 */

#pragma once

#include <algorithm>
#include <iterator>
#include <type_traits>

namespace microtorch {

namespace internal {

// SFINAE (Substitution Failure Is Not An Error)
template <typename I, bool negative_check = false,
          std::enable_if_t<std::is_integral_v<I>, int> = 0>
struct Iterator {
  // declaration for c++ iterator
  using iterator_category = std::input_iterator_tag;
  using value_type = I;
  using difference_type = std::ptrdiff_t;
  using pointer = I*;
  using reference = I&;
  explicit Iterator(I value) : value(value) {}

  I operator*() const { return value; }

  // Prefix increment
  Iterator& operator++() {
    ++value;
    return *this;
  }

  // Suffix increment
  Iterator operator++(int) {
    const auto copy = *this;
    ++*this;
    return copy;
  }

  bool operator==(const Iterator& other) const {
    if constexpr (negative_check) {
      // Range-for loops' end test is `begin != end`, not `begin < end`.
      return (other.value < 0) || value == other.value;
    } else {
      return value == other.value;
    }
  }

  bool operator!=(const Iterator& other) const { return !(*this == other); }

 protected:
  I value;
};

}  // namespace internal

template <typename I, bool negative_check = false,
          std::enable_if_t<std::is_integral_v<I>, int> = 0>
struct IntRange {
 public:
  IntRange(I begin, I end) : begin_(begin), end_(end) {}
  using iterator = internal::Iterator<I, negative_check>;
  iterator begin() const { return begin_; }
  iterator end() const { return end_; }

 private:
  iterator begin_;
  iterator end_;
};

// Creates an integer range for the half-open interval [begin, end)
// If end <= begin, then the range is empty.
// Using the dtype of the `Integer2 end`
template <
    typename Integer1, typename Integer2,
    std::enable_if_t<
        std::is_integral_v<Integer1> && std::is_integral_v<Integer2>, int> = 0>
IntRange<Integer2> irange(Integer1 begin, Integer2 end) {
  return {static_cast<Integer2>(begin),
          std::max(static_cast<Integer2>(begin), end)};
}

// Creates an integer range for the half-open interval [0, end)
template <typename Integer,
          std::enable_if_t<std::is_integral_v<Integer>, int> = 0>
IntRange<Integer, true> irange(Integer end) {
  return {Integer(), end};
}

struct Range {
  Range(int64_t begin, int64_t end) : begin(begin), end(end) {}

  int64_t size() const { return end - begin; }

  Range operator/(int64_t divisor) {
    return Range(begin / divisor, end / divisor);
  }

  int64_t begin;
  int64_t end;
};

}  // namespace microtorch
