/**
 * Copyright (c) 2022-2023 yewentao
 * Licensed under the MIT License.
 */
#pragma once

#include <array>
#include <cstring>
#include <string>
#include <vector>

#include "allocator.hpp"
#include "device.hpp"
#include "macros.hpp"

namespace microtorch {

class Storage {
 public:
  explicit Storage(int64_t nbytes, Device device, const data_t* data = nullptr);
  explicit Storage(const Storage& other) = default;
  explicit Storage(Storage&& other) = default;
  ~Storage() = default;
  Storage& operator=(const Storage& other) = delete;

  data_t operator[](int64_t idx) const;
  data_t& operator[](int64_t idx);
  data_t* data() { return data_ptr_.get(); }
  const Device& device() const { return device_; }
  int64_t nbytes() const { return nbytes_; }

 private:
  int64_t nbytes_;
  Device device_;
  // Note: it seems that shared_ptr is used for controlling cuda memory here
  // But it actually has a deleter for cuda, see allocator.hpp for details.
  std::shared_ptr<data_t> data_ptr_;
};

}  // namespace microtorch
