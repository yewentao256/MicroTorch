#pragma once

#include <vector>

#include "exception.hpp"

namespace microtorch {
struct ArrayRef {
  ArrayRef() {}
  ArrayRef(const ArrayRef&) = default;
  ArrayRef(ArrayRef&&) = default;
  ArrayRef(int64_t size) : data_(size) {}
  ArrayRef(const std::vector<int64_t>& data) : data_(data) {}
  ArrayRef(std::vector<int64_t>&& data) : data_(std::move(data)) {}
  ArrayRef(const std::initializer_list<int64_t>& data) : data_(data) {}
  ArrayRef& operator=(const ArrayRef&) = default;
  ArrayRef& operator=(ArrayRef&&) = default;

  int64_t& operator[](int64_t i) {
    i = i >= 0 ? i : i + this->size();
    TORCH_CHECK(i >= 0 && i < size(), "Index `", i,
                "` is invalid. Index should be a non-negative integer "
                "(including 0) and smaller than size `",
                this->size(), "`.");
    return data_[i];
  }
  const int64_t& operator[](int64_t i) const {
    i = i >= 0 ? i : i + this->size();
    TORCH_CHECK(i >= 0 && i < size(), "Index `", i,
                "` is invalid. Index should be a non-negative integer "
                "(including 0) and smaller than size `",
                this->size(), "`.");
    return data_[i];
  }
  int64_t size() const { return data_.size(); }
  void resize(int64_t s) { data_.resize(s); }
  std::vector<int64_t>& vec() { return data_; }
  const std::vector<int64_t>& vec() const { return data_; }
  int64_t numel() const {
    int64_t result = 1;
    for (auto data : data_) {
      result *= data;
    }
    return result;
  }
  bool operator==(const ArrayRef& other) const { return data_ == other.data_; }

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
  std::vector<int64_t> data_;
};

}  // namespace microtorch