#pragma once

namespace tinytorch {



static void* malloc_host(size_t bytes) { return malloc(bytes); }

static void free_host(void* ptr) { free(ptr); }

static void* malloc_device(size_t bytes) {
  void* ptr;
  cudaMalloc(&ptr, bytes);
  return ptr;
}

static void free_device(void* ptr) {
  cudaFree(ptr);
}

typedef void* (*MallocFunc)(size_t);
typedef void (*FreeFunc)(void*);

class Storage final {
 private:
  MallocFunc malloc_func_;
  FreeFunc free_func_;
  size_t storage_size_;
  void* data_;

 public:
  Storage(MallocFunc malloc_func, FreeFunc free_func, size_t storage_size)
      : malloc_func_(malloc_func),
        free_func_(free_func),
        storage_size_(storage_size) {
    data_ = malloc_func_(storage_size_);
  }

  ~Storage() {
    free_func_(data_);
    data_ = nullptr;
    storage_size_ = 0;
  }

  void* data() { return data_; }
  const void* data() const { return data_; }
  size_t size() const { return storage_size_; }
};

}  // namespace tinytorch