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

}  // namespace tinytorch