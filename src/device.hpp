#pragma once

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace tinytorch {

enum class DeviceType : int8_t {
  CPU = 0,
  CUDA = 1,  // CUDA.
  COMPILE_TIME_MAX_DEVICE_TYPES = 2,
};

struct Device final {
  Device(DeviceType type) : type_(type) {}
  /// Constructs a `Device` from a string description, for convenience.
  Device(const std::string& device_string);

  bool operator==(const Device& other) const {
    return this->type_ == other.type_;
  }

  bool operator!=(const Device& other) const { return !(*this == other); }

  /// Returns the type of device this is.
  DeviceType type() const { return type_; }

  /// Return true if the device is of CUDA type.
  bool is_cuda() const { return type_ == DeviceType::CUDA; }

  /// Return true if the device is of CPU type.
  bool is_cpu() const { return type_ == DeviceType::CPU; }

  /// Same string as returned from operator<<.
  std::string str() const;

  friend std::ostream& operator<<(std::ostream& stream, const Device& device) {
    stream << device.str();
    return stream;
  }

 private:
  DeviceType type_;
};

}  // namespace tinytorch
