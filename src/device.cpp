#include "device.hpp"

#include "exception.hpp"

namespace tinytorch {

Device::Device(const std::string& device_string) : Device(DeviceType::CPU) {
  TORCH_CHECK(!device_string.empty(), "Device string must not be empty");

  if (device_string == "cuda") {
    type_ = DeviceType::CUDA;
  } else if (device_string == "cpu") {
    type_ = DeviceType::CPU;
  } else {
    TORCH_CHECK(false, "Unexpected device string: ", device_string);
  }
}

std::string DeviceTypeName(DeviceType d) {
  switch (d) {
    // I considered instead using ctype::tolower to lower-case the strings
    // on the fly, but this seemed a bit much.
    case DeviceType::CPU:
      return "cpu";
    case DeviceType::CUDA:
      return "cuda";
    default:
      TORCH_CHECK(false, "Unknown device: ", static_cast<int16_t>(d));
  }
}

std::string Device::str() const { return DeviceTypeName(type()); }

}  // namespace tinytorch