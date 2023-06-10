#include "storage.hpp"
#include "exception.hpp"

namespace tinytorch {
Storage::Storage(size_t size)
        : bptr_(Allocator::shared_allocate<Vdata>(size * sizeof(data_t) + sizeof(size_t))),
          dptr_(bptr_->data_) {
    bptr_->version_ = 0;
}

Storage::Storage(const Storage& other, size_t offset)
        : bptr_(other.bptr_),
          dptr_(other.dptr_ + offset) {}

Storage::Storage(size_t size, data_t value)
        : Storage(size) {
    std::memset(dptr_, value, size * sizeof(data_t));
}

Storage::Storage(const data_t* data, size_t size)
        : Storage(size) {
    std::memcpy(dptr_, data, size * sizeof(data_t));
}

}  // namespace tinytorch