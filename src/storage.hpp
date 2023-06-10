#ifndef STORAGE_HPP
#define STORAGE_HPP

#include <array>
#include <string>
#include <vector>
#include <cstring>

#include "allocator.hpp"

namespace tinytorch {

class Storage {
 public:
  explicit Storage(size_t size);
  Storage(const Storage& other, size_t offset);
  Storage(size_t size, data_t value);
  Storage(const data_t* data, size_t size);

  explicit Storage(const Storage& other) = default;
  explicit Storage(Storage&& other) = default;
  ~Storage() = default;
  Storage& operator=(const Storage& other) = delete;

  // inline function
  data_t operator[](size_t idx) const { return dptr_[idx]; }
  data_t& operator[](size_t idx) { return dptr_[idx]; }
  size_t offset(void) const { return dptr_ - bptr_->data_; }
  size_t version(void) const { return bptr_->version_; }
  void increment_version(void) const { bptr_->version_++; }
  data_t* data() { return dptr_; }

 private:
  struct Vdata {
    size_t version_;  // grad version
    data_t data_[1];  // start position of the pointer
  };

  std::shared_ptr<Vdata> bptr_;  // base pointer
  data_t* dptr_;                 // data pointer, offset = dptr_ - bptr_->data_
};

}  // namespace tinytorch

#endif