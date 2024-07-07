/**
 * Copyright (c) 2022-2024 yewentao256
 * Licensed under the MIT License.
 */
#include "storage.hpp"

#include "exception.hpp"

namespace microtorch {

Storage::Storage(int64_t nbytes, Device device, const data_t* data)
    : nbytes_(nbytes),
      device_(device),
      data_ptr_(
          g_allocator_manager.get_allocator(device)->shared_allocate<data_t>(
              nbytes)) {
  if (data) {
#ifdef USE_CUDA
    if (device == Device("cpu")) {
      std::memcpy(data_ptr_.get(), data, nbytes);
    } else {
      cudaMemcpy(data_ptr_.get(), data, nbytes, cudaMemcpyHostToDevice);
    }
#else
    std::memcpy(data_ptr_.get(), data, nbytes);
#endif
  }
}

data_t Storage::operator[](int64_t idx) const {
#ifdef USE_CUDA
  if (device_.is_cuda()) {
    data_t value;
    cudaMemcpy(&value, data_ptr_.get() + idx, sizeof(data_t),
               cudaMemcpyDeviceToHost);
    return value;
  } else {
    return data_ptr_.get()[idx];
  }
#else
  return data_ptr_.get()[idx];
#endif
}

data_t& Storage::operator[](int64_t idx) {
#ifdef USE_CUDA
  TORCH_CHECK(device_.is_cpu(),
              "Non-const indexing into GPU storage is not supported.");
#endif
  return data_ptr_.get()[idx];
}

}  // namespace microtorch