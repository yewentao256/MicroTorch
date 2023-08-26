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
  // class to deallocate the `unique_ptr` with
  class mutable_delete_handler {
   public:
    mutable_delete_handler(size_t size, Allocator* allocator)
        : size_(size), allocator_(allocator) {}
    void operator()(void* ptr) { allocator_->deallocate(ptr, size_); }

   private:
    size_t size_;
    Allocator* allocator_;
  };

  template <typename T>
  class immutable_delete_handler {
   public:
    immutable_delete_handler(Allocator* allocator) : allocator_(allocator) {}
    void operator()(void* ptr) {
      static_cast<T*>(ptr)->~T();
      allocator_->deallocate(ptr, sizeof(T));
    }

   private:
    Allocator* allocator_;
  };

  template <typename T>
  using MutableUniquePtr = std::unique_ptr<T, mutable_delete_handler>;

  template <typename T>
  using ImmutableUniquePtr = std::unique_ptr<T, immutable_delete_handler<T>>;

  template <typename T>
  std::shared_ptr<T> shared_allocate(size_t nbytes) {
    /* allocate the nbytes, return shared_ptr */
    void* raw_ptr = allocate(nbytes);
    return std::shared_ptr<T>(static_cast<T*>(raw_ptr),
                              mutable_delete_handler(nbytes, this));
  }

  template <typename T>
  MutableUniquePtr<T> unique_allocate(size_t nbytes) {
    /* allocate the nbytes, return unique_ptr */
    void* raw_ptr = allocate(nbytes);
    return MutableUniquePtr<T>(static_cast<T*>(raw_ptr),
                               mutable_delete_handler(nbytes, this));
  }

  template <typename T, typename... Args>
  std::shared_ptr<T> shared_construct(Args&&... args) {
    /* construct the object, return shared_ptr */
    void* raw_ptr = allocate(sizeof(T));
    new (raw_ptr) T(std::forward<Args>(args)...);
    return std::shared_ptr<T>(static_cast<T*>(raw_ptr),
                              immutable_delete_handler<T>(this));
  }

  template <typename T, typename... Args>
  ImmutableUniquePtr<T> unique_construct(Args&&... args) {
    /* construct the object, return unique_ptr */
    void* raw_ptr = allocate(sizeof(T));
    new (raw_ptr) T(std::forward<Args>(args)...);
    return ImmutableUniquePtr<T>(static_cast<T*>(raw_ptr),
                                 immutable_delete_handler<T>(this));
  }

  bool all_clear(void) {
    return allocate_memory_size_ == deallocate_memory_size_;
  }

  virtual ~Allocator() {}

 protected:
  virtual void* do_allocate(size_t size) = 0;
  virtual void do_deallocate(void* ptr) = 0;

  void* allocate(size_t size) {
    auto iter = cache_.find(size);
    void* res;
    if (iter != cache_.end()) {
      // Found pointer that can be reused.
      res = iter->second.release();
      cache_.erase(iter);
    } else {
      res = do_allocate(size);
      TORCH_CHECK(res, "failed to allocate", size, "memory.");
    }
    allocate_memory_size_ += size;
    return res;
  }
  void deallocate(void* ptr, size_t size) {
    deallocate_memory_size_ += size;
    cache_.emplace(size,
                   std::unique_ptr<void, std::function<void(void*)>>(
                       ptr, [this](void* ptr) { this->do_deallocate(ptr); }));
  }

  size_t allocate_memory_size_ = 0;
  size_t deallocate_memory_size_ = 0;

  // virtual void free_deletor(void* ptr) = 0;
  /* cache_ saves all of the pointers that have been deallocated.
     So we can reuse it instead of malloc again */
  std::multimap<size_t, std::unique_ptr<void, std::function<void(void*)>>>
      cache_;
};

class CPUAllocator : public Allocator {
 public:
  void* do_allocate(size_t size) override { return std::malloc(size); }

  void do_deallocate(void* ptr) override { std::free(ptr); }

  ~CPUAllocator() override {
    for (auto& pair : cache_) {
      do_deallocate(pair.second.release());
    }
    cache_.clear();
  }
};

#ifdef USE_CUDA
class CUDAAllocator : public Allocator {
 public:
  void* do_allocate(size_t size) override {
    void* ptr;
    cudaMalloc(&ptr, size);
    return ptr;
  }

  void do_deallocate(void* ptr) override { cudaFree(ptr); }

  ~CUDAAllocator() override {
    for (auto& pair : cache_) {
      do_deallocate(pair.second.release());
    }
    cache_.clear();
  }
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
    return cpu_allocator_.get();
  }

 private:
  std::unique_ptr<CPUAllocator> cpu_allocator_;
#ifdef USE_CUDA
  std::unique_ptr<CUDAAllocator> cuda_allocator_;
#endif
};

extern AllocatorManager g_allocator_manager;

}  // namespace microtorch
