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
  DeviceType type() const { return type_; }
  bool is_cuda() const { return type_ == DeviceType::CUDA; }
  bool is_cpu() const { return type_ == DeviceType::CPU; }

  std::string str() const;
  operator std::string() const {
    // implicitly convert
    return str();
  }

  friend std::ostream& operator<<(std::ostream& stream, const Device& device) {
    stream << device.str();
    return stream;
  }

 private:
  DeviceType type_;
};

}  // namespace tinytorch
