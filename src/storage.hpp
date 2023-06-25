#pragma once

#include <array>
#include <cstring>
#include <string>
#include <vector>

#include "allocator.hpp"
#include "device.hpp"

namespace tinytorch {

using data_t = float;

class Storage {
 public:
  explicit Storage(size_t nbytes, Device device);
  // Storage(const Storage& other, Device device);
  // Storage(size_t nbytes, data_t value, Device device);
  // Storage(const data_t* data, size_t nbytes, Device device);

  explicit Storage(const Storage& other) = default;
  explicit Storage(Storage&& other) = default;
  ~Storage() = default;
  Storage& operator=(const Storage& other) = delete;

  data_t operator[](size_t idx) const { return data_ptr_.get()[idx]; }
  data_t& operator[](size_t idx) { return data_ptr_.get()[idx]; }
  size_t version(void) const { return version_; }
  void increment_version(void) { version_ += 1; }
  data_t* data() { return data_ptr_.get(); }
  const Device& device() const { return device_; }
  size_t nbytes() const { return nbytes_; }

 private:
  size_t nbytes_;
  Device device_;
  size_t version_ = 0;
  std::shared_ptr<data_t> data_ptr_;
};

}  // namespace tinytorch
