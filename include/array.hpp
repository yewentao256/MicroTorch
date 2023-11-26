/**
 * Copyright (c) 2022-2023 yewentao
 * Licensed under the MIT License.
 */
#pragma once

#include <type_traits>
#include <vector>

#include "exception.hpp"

namespace microtorch {

template <typename T>
struct ArrayRef {
  ArrayRef() {}
  ArrayRef(const ArrayRef&) = default;
  ArrayRef(ArrayRef&&) = default;
  ArrayRef(int64_t size) : data_(size) {}
  ArrayRef(const std::vector<T>& data) : data_(data) {}
  ArrayRef(std::vector<T>&& data) : data_(std::move(data)) {}
  ArrayRef(const std::initializer_list<T>& data) : data_(data) {}
  ArrayRef(T* start, T* end) : data_(start, end) {}  // accepting for 2 pointers
  ArrayRef& operator=(const ArrayRef&) = default;
  ArrayRef& operator=(ArrayRef&&) = default;

  T& operator[](int64_t i) {
    i = i >= 0 ? i : i + this->size();
    TORCH_CHECK(i >= 0 && i < size(), "Index `", i,
                "` is invalid. Index should be a non-negative integer "
                "(including 0) and smaller than size `",
                this->size(), "`.");
    return data_[i];
  }
  const T& operator[](int64_t i) const {
    i = i >= 0 ? i : i + this->size();
    TORCH_CHECK(i >= 0 && i < size(), "Index `", i,
                "` is invalid. Index should be a non-negative integer "
                "(including 0) and smaller than size `",
                this->size(), "`.");
    return data_[i];
  }
  int64_t size() const { return data_.size(); }
  void resize(int64_t s, const T& value = T()) { data_.resize(s, value); }
  std::vector<T>& vec() { return data_; }
  const std::vector<T>& vec() const { return data_; }
  T* data() { return data_.data(); }
  const T* data() const { return data_.data(); }
  int64_t numel() const {
    static_assert(std::is_integral_v<T>,
                  "numel() is valid only for integral types.");
    int64_t result = 1;
    for (auto data : data_) {
      result *= data;
    }
    return result;
  }
  bool operator==(const ArrayRef& other) const { return data_ == other.data_; }
  bool operator!=(const ArrayRef& other) const { return !(*this == other); }

  void push_back(const T& value) { data_.push_back(value); }
  void push_back(T&& value) { data_.push_back(std::move(value)); }

  auto begin() { return data_.begin(); }
  auto begin() const { return data_.cbegin(); }
  auto end() { return data_.end(); }
  auto end() const { return data_.cend(); }
  auto rbegin() { return data_.rbegin(); }
  auto rbegin() const { return data_.crbegin(); }
  auto rend() { return data_.rend(); }
  auto rend() const { return data_.crend(); }

  auto erase(typename std::vector<T>::const_iterator position) {
    return data_.erase(position);
  }
  auto erase(typename std::vector<T>::const_iterator first,
             typename std::vector<T>::const_iterator last) {
    return data_.erase(first, last);
  }
  auto insert(typename std::vector<T>::const_iterator position,
              const T& value) {
    return data_.insert(position, value);
  }

  bool empty() const { return data_.empty(); }

  friend std::ostream& operator<<(std::ostream& os, const ArrayRef& arr) {
    os << "[";
    if (!arr.data_.empty()) {
      os << arr.data_[0];
      for (size_t i = 1; i < arr.data_.size(); ++i) {
        os << ", " << arr.data_[i];
      }
    }
    os << "]";
    return os;
  }

 private:
  std::vector<T> data_;
};

using IntArrayRef = ArrayRef<int64_t>;
using PtrArrayRef = ArrayRef<char*>;

}  // namespace microtorch