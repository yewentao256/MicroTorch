//===--- MArrayRef.h - Array Reference Wrapper -------------------*- C++
//-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <array>
#include <iterator>
#include <vector>

#include "exception.hpp"
#include "smallVector.hpp"

namespace microtorch {
// MArrayRef - Represent a constant reference to an array (0 or more elements
// consecutively in memory), i.e. a start pointer and a length.  It allows
// various APIs to take consecutive elements easily and conveniently.
//
// This class does not own the underlying data, it is expected to be used in
// situations where the data resides in some other buffer, whose lifetime
// extends past that of the MArrayRef.
//
// Trivially copyable, so it should be passed by value.
template <typename T>
class MArrayRef final {
 public:
  using iterator = const T*;
  using const_iterator = const T*;
  using size_type = size_t;
  using value_type = T;

  using reverse_iterator = std::reverse_iterator<iterator>;

 private:
  // The start of the array, in an external buffer.
  const T* Data;

  // The number of elements.
  size_type Length;

  void debugCheckNullptrInvariant() {
    TORCH_INTERNAL_ASSERT(Data != nullptr || Length == 0);
  }

 public:
  // Construct an empty MArrayRef.
  constexpr MArrayRef() : Data(nullptr), Length(0) {}

  // Construct an MArrayRef from a single element.
  constexpr explicit MArrayRef(const T& OneElt) : Data(&OneElt), Length(1) {}

  // Construct an MArrayRef from a pointer and length.
  MArrayRef(const T* data, size_t length) : Data(data), Length(length) {
    debugCheckNullptrInvariant();
  }

  // Construct an MArrayRef from a range.
  MArrayRef(const T* begin, const T* end) : Data(begin), Length(end - begin) {
    debugCheckNullptrInvariant();
  }

  // Construct an MArrayRef from a SmallVector. This is templated in order to
  // avoid instantiating SmallVectorTemplateCommon<T> whenever we
  // copy-construct an MArrayRef.
  template <typename U>
  MArrayRef(const SmallVectorTemplateCommon<T, U>& Vec)
      : Data(Vec.data()), Length(Vec.size()) {
    debugCheckNullptrInvariant();
  }

  template <typename Container,
            typename = std::enable_if_t<std::is_same<
                std::remove_const_t<decltype(std::declval<Container>().data())>,
                T*>::value>>
  MArrayRef(const Container& container)
      : Data(container.data()), Length(container.size()) {
    debugCheckNullptrInvariant();
  }

  // Construct an MArrayRef from a std::vector.
  // The enable_if stuff here makes sure that this isn't used for
  // std::vector<bool>, because MArrayRef can't work on a std::vector<bool>
  // bitfield.
  template <typename A>
  MArrayRef(const std::vector<T, A>& Vec)
      : Data(Vec.data()), Length(Vec.size()) {
    static_assert(!std::is_same<T, bool>::value,
                  "MArrayRef<bool> cannot be constructed from a "
                  "std::vector<bool> bitfield.");
  }

  // Construct an MArrayRef from a std::array
  template <size_t N>
  constexpr MArrayRef(const std::array<T, N>& Arr)
      : Data(Arr.data()), Length(N) {}

  // Construct an MArrayRef from a C array.
  template <size_t N>
  constexpr MArrayRef(const T (&Arr)[N]) : Data(Arr), Length(N) {}

  // Construct an MArrayRef from a std::initializer_list.
  constexpr MArrayRef(const std::initializer_list<T>& Vec)
      : Data(std::begin(Vec) == std::end(Vec) ? static_cast<T*>(nullptr)
                                              : std::begin(Vec)),
        Length(Vec.size()) {}

  constexpr iterator begin() const { return Data; }
  constexpr iterator end() const { return Data + Length; }

  // These are actually the same as iterator, since MArrayRef only
  // gives you const iterators.
  constexpr const_iterator cbegin() const { return Data; }
  constexpr const_iterator cend() const { return Data + Length; }

  constexpr reverse_iterator rbegin() const { return reverse_iterator(end()); }
  constexpr reverse_iterator rend() const { return reverse_iterator(begin()); }

  // empty - Check if the array is empty.
  constexpr bool empty() const { return Length == 0; }

  constexpr const T* data() const { return Data; }

  // size - Get the array size.
  constexpr size_t size() const { return Length; }

  // front - Get the first element.
  const T& front() const {
    TORCH_CHECK(!empty(),
                "MArrayRef: attempted to access front() of empty list");
    return Data[0];
  }

  // back - Get the last element.
  const T& back() const {
    TORCH_CHECK(!empty(),
                "MArrayRef: attempted to access back() of empty list");
    return Data[Length - 1];
  }

  // equals - Check for element-wise equality.
  constexpr bool equals(MArrayRef RHS) const {
    return Length == RHS.Length && std::equal(begin(), end(), RHS.begin());
  }

  // slice(n, m) - Take M elements of the array starting at element N
  MArrayRef<T> slice(size_t N, size_t M) const {
    TORCH_CHECK(N + M <= size(), "MArrayRef: invalid slice, N = ", N,
                "; M = ", M, "; size = ", size());
    return MArrayRef<T>(data() + N, M);
  }

  // slice(n) - Chop off the first N elements of the array.
  MArrayRef<T> slice(size_t N) const {
    TORCH_CHECK(N <= size(), "MArrayRef: invalid slice, N = ", N,
                "; size = ", size());
    return slice(N, size() - N);
  }

  constexpr const T& operator[](size_t Index) const { return Data[Index]; }

  // Vector compatibility
  const T& at(size_t Index) const {
    TORCH_CHECK(Index < Length, "MArrayRef: invalid index Index = ", Index,
                "; Length = ", Length);
    return Data[Index];
  }

  // Disallow accidental assignment from a temporary.
  //
  // The declaration here is extra complicated so that "MArrayRef = {}"
  // continues to select the move assignment operator.
  template <typename U>
  typename std::enable_if<std::is_same<U, T>::value, MArrayRef<T>>::type&
  operator=(U&& Temporary) = delete;

  // Disallow accidental assignment from a temporary.
  //
  // The declaration here is extra complicated so that "MArrayRef = {}"
  // continues to select the move assignment operator.
  template <typename U>
  typename std::enable_if<std::is_same<U, T>::value, MArrayRef<T>>::type&
  operator=(std::initializer_list<U>) = delete;

  std::vector<T> vec() const { return std::vector<T>(Data, Data + Length); }
};

template <typename T>
std::ostream& operator<<(std::ostream& out, MArrayRef<T> list) {
  int i = 0;
  out << "[";
  for (const auto& e : list) {
    if (i++ > 0) out << ", ";
    out << e;
  }
  out << "]";
  return out;
}

// MArrayRef Convenience constructors

// Construct an MArrayRef from a single element.
template <typename T>
MArrayRef<T> makeArrayRef(const T& OneElt) {
  return OneElt;
}

// Construct an MArrayRef from a pointer and length.
template <typename T>
MArrayRef<T> makeArrayRef(const T* data, size_t length) {
  return MArrayRef<T>(data, length);
}

// Construct an MArrayRef from a range.
template <typename T>
MArrayRef<T> makeArrayRef(const T* begin, const T* end) {
  return MArrayRef<T>(begin, end);
}

// Construct an MArrayRef from a SmallVector.
template <typename T>
MArrayRef<T> makeArrayRef(const SmallVectorImpl<T>& Vec) {
  return Vec;
}

// Construct an MArrayRef from a SmallVector.
template <typename T, unsigned N>
MArrayRef<T> makeArrayRef(const SmallVector<T, N>& Vec) {
  return Vec;
}

// Construct an MArrayRef from a std::vector.
template <typename T>
MArrayRef<T> makeArrayRef(const std::vector<T>& Vec) {
  return Vec;
}

// Construct an MArrayRef from a std::array.
template <typename T, std::size_t N>
MArrayRef<T> makeArrayRef(const std::array<T, N>& Arr) {
  return Arr;
}

// Construct an MArrayRef from an MArrayRef (no-op) (const)
template <typename T>
MArrayRef<T> makeArrayRef(const MArrayRef<T>& Vec) {
  return Vec;
}

// Construct an MArrayRef from an MArrayRef (no-op)
template <typename T>
MArrayRef<T>& makeArrayRef(MArrayRef<T>& Vec) {
  return Vec;
}

// Construct an MArrayRef from a C array.
template <typename T, size_t N>
MArrayRef<T> makeArrayRef(const T (&Arr)[N]) {
  return MArrayRef<T>(Arr);
}

// WARNING: Template instantiation will NOT be willing to do an implicit
// conversions to get you to an microtorch::MArrayRef, which is why we need so
// many overloads.

template <typename T>
bool operator==(microtorch::MArrayRef<T> a1, microtorch::MArrayRef<T> a2) {
  return a1.equals(a2);
}

template <typename T>
bool operator!=(microtorch::MArrayRef<T> a1, microtorch::MArrayRef<T> a2) {
  return !a1.equals(a2);
}

template <typename T>
bool operator==(const std::vector<T>& a1, microtorch::MArrayRef<T> a2) {
  return microtorch::MArrayRef<T>(a1).equals(a2);
}

template <typename T>
bool operator!=(const std::vector<T>& a1, microtorch::MArrayRef<T> a2) {
  return !microtorch::MArrayRef<T>(a1).equals(a2);
}

template <typename T>
bool operator==(microtorch::MArrayRef<T> a1, const std::vector<T>& a2) {
  return a1.equals(microtorch::MArrayRef<T>(a2));
}

template <typename T>
bool operator!=(microtorch::MArrayRef<T> a1, const std::vector<T>& a2) {
  return !a1.equals(microtorch::MArrayRef<T>(a2));
}

using IntMArrayRef = MArrayRef<int64_t>;

}  // namespace microtorch
