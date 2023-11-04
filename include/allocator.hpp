/**
 * Copyright (c) 2022-2023 yewentao
 * Licensed under the MIT License.
 */
#pragma once

#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include "device.hpp"
#include "exception.hpp"

namespace microtorch {

class Allocator {
 public:
  class mutable_delete_handler {
    // class to deallocate
   public:
    mutable_delete_handler(int64_t size, Allocator* allocator)
        : size_(size), allocator_(allocator) {}
    void operator()(void* ptr) { allocator_->deallocate(ptr, size_); }

   private:
    int64_t size_;
    Allocator* allocator_;
  };

  template <typename T>
  using MutableUniquePtr = std::unique_ptr<T, mutable_delete_handler>;

  template <typename T>
  std::shared_ptr<T> shared_allocate(int64_t nbytes) {
    /* allocate the nbytes, return shared_ptr */
    void* raw_ptr = allocate(nbytes);
    return std::shared_ptr<T>(static_cast<T*>(raw_ptr),
                              mutable_delete_handler(nbytes, this));
  }

  template <typename T>
  MutableUniquePtr<T> unique_allocate(int64_t nbytes) {
    /* allocate the nbytes, return unique_ptr */
    void* raw_ptr = allocate(nbytes);
    return MutableUniquePtr<T>(static_cast<T*>(raw_ptr),
                               mutable_delete_handler(nbytes, this));
  }

  void clear() {
    for (auto& pair : cache_) {
      do_deallocate(pair.second.release());
    }
    cache_.clear();
    allocate_memory_size_ = 0;
    deallocate_memory_size_ = 0;
  }

  bool check_all_clear(void) {
    return allocate_memory_size_ == deallocate_memory_size_;
  }

  virtual ~Allocator() = default;

 protected:
  virtual void* do_allocate(int64_t size) = 0;
  virtual void do_deallocate(void* ptr) = 0;

  void* allocate(int64_t size) {
    auto iter = cache_.find(size);
    void* res;
    if (iter != cache_.end()) {
      // Found pointer that can be reused.
      res = iter->second.release();
      cache_.erase(iter);
    } else {
      res = do_allocate(size);
      TORCH_CHECK(res, "failed to allocate `", size, "` of memory.");
    }
    allocate_memory_size_ += size;
    return res;
  }
  void deallocate(void* ptr, int64_t size) {
    deallocate_memory_size_ += size;
    cache_.emplace(size,
                   std::unique_ptr<void, std::function<void(void*)>>(
                       ptr, [this](void* ptr) { this->do_deallocate(ptr); }));
  }

  int64_t allocate_memory_size_ = 0;
  int64_t deallocate_memory_size_ = 0;

  /* cache_ saves all of the pointers that have been deallocated.
     So we can reuse it instead of malloc again */
  std::multimap<int64_t, std::unique_ptr<void, std::function<void(void*)>>>
      cache_;
};

class CPUAllocator : public Allocator {
 public:
  void* do_allocate(int64_t size) override { return std::malloc(size); }
  void do_deallocate(void* ptr) override { std::free(ptr); }
  ~CPUAllocator() { 
    // Note: we can't call clear() in the deconstructor of `Allocator` class,
    // because the clear() calls the pure virtual function `do_deallocate`.
    // The destructor of the derived class is called before the base class,
    // so you will get a `Pure virtual function called!` error.
    // Write it here is safe.
    clear(); }
};

#ifdef USE_CUDA
class CUDAAllocator : public Allocator {
 public:
  void* do_allocate(int64_t size) override {
    void* ptr;
    cudaMalloc(&ptr, size);
    return ptr;
  }

  void do_deallocate(void* ptr) override { cudaFree(ptr); }
  ~CUDAAllocator() override { clear(); }
};
#endif

class AllocatorManager {
 public:
  AllocatorManager() {
    cpu_allocator_ = std::make_unique<CPUAllocator>();
#ifdef USE_CUDA
    cuda_allocator_ = std::make_unique<CUDAAllocator>();
#endif
  }
  Allocator* get_allocator(Device d) {
#ifdef USE_CUDA
    if (d.is_cpu()) {
      return cpu_allocator_.get();
    } else {
      return cuda_allocator_.get();
    }
#endif
    TORCH_CHECK(
        d.is_cpu(),
        "Only supports get a cpu allocator when not compiled with CUDA.");
    return cpu_allocator_.get();
  }
  void reset_allocators() {
    cpu_allocator_->clear();
#ifdef USE_CUDA
    cuda_allocator_->clear();
#endif
  }

 private:
  std::unique_ptr<CPUAllocator> cpu_allocator_;
#ifdef USE_CUDA
  std::unique_ptr<CUDAAllocator> cuda_allocator_;
#endif
};

extern AllocatorManager g_allocator_manager;

}  // namespace microtorch
