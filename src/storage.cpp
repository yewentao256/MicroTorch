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
                    if (device == Device("cpu")){
                        std::memcpy(data_ptr_.get(), data, nbytes);
                    }
                    else {
                        cudaMemcpy(data_ptr_.get(), data, nbytes,
                        cudaMemcpyDeviceToHost);
                    }
                    #else
                        std::memcpy(data_ptr_.get(), data, nbytes);
                    #endif
                }
              }

// Storage::Storage(const Storage& other, Device device)
//     : device_(device), data_ptr_(other.data_ptr_), nbytes_(other.nbytes()) {}

}  // namespace tinytorch