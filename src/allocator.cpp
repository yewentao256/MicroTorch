#include "allocator.hpp"

namespace tinytorch {

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