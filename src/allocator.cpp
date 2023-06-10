#include "allocator.hpp"

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

size_t Allocator::allocate_memory_size_;
size_t Allocator::deallocate_memory_size_;

void* Allocator::allocate(size_t size) {
  auto& cache = singleton().cache_;
  auto iter = cache.find(size);
  void* res;
  if (iter != cache.end()) {
    // Found pointer that can be reused.
    // release unique_ptr, get the raw pointer
    res = iter->second.release();
    cache.erase(iter);
  } else {
    res = std::malloc(size);
    TORCH_CHECK(res, "failed to allocate", size, "memory.");
  }
  allocate_memory_size_ += size;
  return res;
}

void Allocator::deallocate(void* ptr, size_t size) {
  deallocate_memory_size_ += size;
  singleton().cache_.emplace(size, ptr);
}

bool Allocator::all_clear() {
  return allocate_memory_size_ == deallocate_memory_size_;
}

}  // namespace tinytorch