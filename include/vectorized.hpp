/**
 * Copyright (c) 2022-2023 yewentao
 * Licensed under the MIT License.
 */
#pragma once
#include "loops.hpp"

namespace microtorch {

constexpr int vectorWidth = 32;

// Vectorized is an array of <T>, aligning to vectorWidth. This is designed for
// better performance in SIMD(Single Instruction Multiple Data)
template <class T>
struct Vectorized {
 private:
  __attribute__((aligned(32))) T values[vectorWidth / sizeof(T)];

 public:
  using value_type = T;
  using size_type = int;
  static constexpr size_type size_T = sizeof(T);
  static constexpr size_type size() { return vectorWidth / size_T; }
  Vectorized() : values{static_cast<T>(0)} {}
  Vectorized(T val) {
    for (int i = 0; i != size(); i++) {
      values[i] = val;
    }
  }
  template <typename... Args,
            typename = std::enable_if_t<(sizeof...(Args) == size())>>
  Vectorized(Args... vals) : values{vals...} {}
  // This also implies const T& operator[](int idx) const
  inline operator const T *() const { return values; }
  // This also implies T& operator[](int idx)
  inline operator T *() { return values; }
  static Vectorized<T> loadu(const void *ptr) {
    Vectorized vector;
    std::memcpy(vector.values, ptr, vectorWidth);
    return vector;
  }
  static Vectorized<T> loadu(const void *ptr, int64_t count) {
    Vectorized vector;
    std::memcpy(vector.values, ptr, count * sizeof(T));
    return vector;
  }
  void store(void *ptr, int count = size()) const {
    std::memcpy(ptr, values, count * sizeof(T));
  }

 private:
  template <typename Op>
  inline Vectorized<T> binary_pred(const Vectorized<T> &other, Op op) const {
    // All bits are set to 1 if the pred is true, otherwise 0.
    Vectorized<T> vector;
    for (int64_t i = 0; i != size(); i++) {
      if (op(values[i], other.values[i])) {
        std::memset(static_cast<void *>(vector.values + i), 0xFF, sizeof(T));
      } else {
        std::memset(static_cast<void *>(vector.values + i), 0, sizeof(T));
      }
    }
    return vector;
  }

 public:
  Vectorized<T> operator==(const Vectorized<T> &other) const {
    return binary_pred(other, std::equal_to<T>());
  }
  Vectorized<T> operator!=(const Vectorized<T> &other) const {
    return binary_pred(other, std::not_equal_to<T>());
  }
  Vectorized<T> operator>=(const Vectorized<T> &other) const {
    return binary_pred(other, std::greater_equal<T>());
  }
  Vectorized<T> operator<=(const Vectorized<T> &other) const {
    return binary_pred(other, std::less_equal<T>());
  }
  Vectorized<T> operator>(const Vectorized<T> &other) const {
    return binary_pred(other, std::greater<T>());
  }
  Vectorized<T> operator<(const Vectorized<T> &other) const {
    return binary_pred(other, std::less<T>());
  }

 private:
  template <typename Op>
  inline Vectorized<T> binary_pred_bool(const Vectorized<T> &other,
                                        Op op) const {
    // 1 if the pred is true, otherwise 0.
    Vectorized<T> vector;
    for (int i = 0; i != size(); ++i) {
      vector[i] = static_cast<T>(op(values[i], other.values[i]));
    }
    return vector;
  }
};

template <typename VT>
struct VecHoldType { using hold_type = typename VT::value_type; };

template <typename VT>
using vechold_type = typename VecHoldType<VT>::hold_type;

template <class T>
Vectorized<T> inline operator+(const Vectorized<T> &a, const Vectorized<T> &b) {
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = a[i] + b[i];
  }
  return c;
}

// Load vector from a smaller type (more elements) to a larger type (fewer
// elements), reducing neighboring elements until it fits into the vector size.
template <typename acc_t, typename scalar_t, typename F>
Vectorized<acc_t> load_reduce_vec(const scalar_t *data, F reduce, acc_t ident) {
  using vec_t = Vectorized<scalar_t>;
  using vacc_t = Vectorized<acc_t>;
  static_assert(vacc_t::size() <= vec_t::size(), "");
  const auto val = vec_t::loadu(data);
  alignas(64) std::array<scalar_t, vec_t::size()> values;
  val.store(values.data());

  constexpr int vstride = vec_t::size() / vacc_t::size();
  alignas(64) std::array<acc_t, vacc_t::size()> acc;
  acc.fill(ident);
  for (const auto k : irange(vstride)) {
    for (const auto i : irange(vacc_t::size())) {
      acc[i] = reduce(acc[i], values[i * vstride + k]);
    }
  }

  return vacc_t::loadu(acc.data());
}


template <typename T>
inline Vectorized<T>& operator += (Vectorized<T>& a, const Vectorized<T>& b) {
  a = a + b;
  return a;
}
template <typename T>
inline Vectorized<T>& operator -= (Vectorized<T>& a, const Vectorized<T>& b) {
  a = a - b;
  return a;
}
template <typename T>
inline Vectorized<T>& operator /= (Vectorized<T>& a, const Vectorized<T>& b) {
  a = a / b;
  return a;
}
template <typename T>
inline Vectorized<T>& operator %= (Vectorized<T>& a, const Vectorized<T>& b) {
  a = a % b;
  return a;
}
template <typename T>
inline Vectorized<T>& operator *= (Vectorized<T>& a, const Vectorized<T>& b) {
  a = a * b;
  return a;
}

} // namespace microtorch
