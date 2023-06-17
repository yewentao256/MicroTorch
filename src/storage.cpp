#include "storage.hpp"
#include "exception.hpp"

namespace tinytorch {
Storage::Storage(size_t size, Device device)
    : device_(device),
      bptr_(g_allocator_manager.get_allocator(device)->shared_allocate<Vdata>(
          size * sizeof(data_t) + sizeof(size_t))),
      dptr_(bptr_->data_),
      size_(size) {
  bptr_->version_ = 0;
}

Storage::Storage(const Storage& other, size_t offset, Device device)
    : device_(device), bptr_(other.bptr_), dptr_(other.dptr_ + offset), size_(other.size()) {}

Storage::Storage(size_t size, data_t value, Device device) : Storage(size, device) {
  std::memset(dptr_, value, size * sizeof(data_t));
}

Storage::Storage(const data_t* data, size_t size, Device device) : Storage(size, device) {
  std::memcpy(dptr_, data, size * sizeof(data_t));
}

}  // namespace tinytorch