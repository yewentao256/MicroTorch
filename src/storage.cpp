#include "storage.hpp"

#include "exception.hpp"

namespace tinytorch {
Storage::Storage(size_t nbytes, Device device)
    : nbytes_(nbytes),
      device_(device),
      data_ptr_(
          g_allocator_manager.get_allocator(device)->shared_allocate<data_t>(
              nbytes)) {}

// Storage::Storage(const Storage& other, Device device)
//     : device_(device), data_ptr_(other.data_ptr_), nbytes_(other.nbytes()) {}

// Storage::Storage(size_t nbytes, data_t value, Device device)
//     : Storage(nbytes, device) {
//   std::memset(data_ptr_.get(), value, nbytes * sizeof(data_t));
// }

// Storage::Storage(const data_t* data, size_t nbytes, Device device)
//     : Storage(nbytes, device) {
//   std::memcpy(data_ptr_.get(), data, nbytes * sizeof(data_t));
// }

}  // namespace tinytorch