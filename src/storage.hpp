#ifndef STORAGE_HPP
#define STORAGE_HPP

#include <array>
#include <cstring>
#include <string>
#include <vector>

#include "allocator2.hpp"
#include "device.hpp"

namespace tinytorch {

using data_t = float;

class Storage {
 public:
  explicit Storage(size_t size, Device device = Device("cpu"));
  Storage(const Storage& other, size_t offset, Device device = Device("cpu"));
  Storage(size_t size, data_t value, Device device = Device("cpu"));
  Storage(const data_t* data, size_t size, Device device = Device("cpu"));

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
  Device& device() { return device_; }
  size_t size() const { return size_; }

 private:
  Device device_;
  struct Vdata {
    size_t version_;  // grad version
    data_t data_[1];  // start position of the pointer
  };

  std::shared_ptr<Vdata> bptr_;  // base pointer

  data_t* dptr_;  // data pointer, offset = dptr_ - bptr_->data_

  size_t size_;  // TODO: do we really need this?
};

}  // namespace tinytorch

#endif