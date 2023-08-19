#include "storage.hpp"

#include "exception.hpp"

namespace tinytorch {

Storage::Storage(size_t nbytes, Device device, const data_t* data)
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
      cudaMemcpy(data_ptr_.get(), data, nbytes, cudaMemcpyDeviceToHost);
    }
#else
    std::memcpy(data_ptr_.get(), data, nbytes);
#endif
  }
}

data_t Storage::operator[](size_t idx) const {
#ifdef USE_CUDA
  if (device_.is_cuda()) {
    data_t value;
    cudaMemcpy(&value, data_ptr_.get() + idx, sizeof(data_t), cudaMemcpyDeviceToHost);
    return value;
  } else {
    return data_ptr_.get()[idx];
  }
#else
  return data_ptr_.get()[idx];
#endif
}

data_t& Storage::operator[](size_t idx) {
#ifdef USE_CUDA
  TORCH_CHECK(device_.is_cpu(),
              "Non-const indexing into GPU storage is not supported.");
#endif
  return data_ptr_.get()[idx];
}

// Storage::Storage(const Storage& other, Device device)
//     : device_(device), data_ptr_(other.data_ptr_), nbytes_(other.nbytes()) {}

}  // namespace tinytorch