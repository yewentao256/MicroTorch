#pragma once

#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "exception.hpp"

namespace tinytorch {

class Allocator {
 public:
  // class to deallocate the `unique_ptr`
  class trivial_delete_handler {
   public:
    trivial_delete_handler(size_t size_) : size(size_) {}
    void operator()(void* ptr) { deallocate(ptr, size); }

   private:
    size_t size;
  };

  template <typename T>
  class nontrivial_delete_handler {
   public:
    void operator()(void* ptr) {
      static_cast<T*>(ptr)->~T();
      deallocate(ptr, sizeof(T));
    }
  };

  template <typename T>
  using TrivialUniquePtr = std::unique_ptr<T, trivial_delete_handler>;

  template <typename T>
  using NontrivialUniquePtr = std::unique_ptr<T, nontrivial_delete_handler<T>>;

  // I know it's weird here. The type has been already passed in as T, but the
  // function parameter still need the number of bytes, instead of objects.
  // And their relationship is
  //          nbytes = nobjects * sizeof(T).
  // Check what I do in "tensor/storage.cpp", and you'll understand.
  // Or maybe changing the parameter here and doing some extra work in
  // "tensor/storage.cpp" is better.
  template <typename T>
  static std::shared_ptr<T> shared_allocate(size_t nbytes) {
    /* allocate the nbytes, return shared_ptr */
    void* raw_ptr = allocate(nbytes);
    return std::shared_ptr<T>(static_cast<T*>(raw_ptr),
                              trivial_delete_handler(nbytes));
  }

  template <typename T>
  static TrivialUniquePtr<T> unique_allocate(size_t nbytes) {
    /* allocate the nbytes, return unique_ptr */
    void* raw_ptr = allocate(nbytes);
    return TrivialUniquePtr<T>(static_cast<T*>(raw_ptr),
                               trivial_delete_handler(nbytes));
  }

  template <typename T, typename... Args>
  static std::shared_ptr<T> shared_construct(Args&&... args) {
    /* construct the object, return shared_ptr */
    void* raw_ptr = allocate(sizeof(T));
    new (raw_ptr) T(std::forward<Args>(args)...);
    return std::shared_ptr<T>(static_cast<T*>(raw_ptr),
                              nontrivial_delete_handler<T>());
  }

  template <typename T, typename... Args>
  static NontrivialUniquePtr<T> unique_construct(Args&&... args) {
    /* construct the object, return unique_ptr */
    void* raw_ptr = allocate(sizeof(T));
    new (raw_ptr) T(std::forward<Args>(args)...);
    return NontrivialUniquePtr<T>(static_cast<T*>(raw_ptr),
                                  nontrivial_delete_handler<T>());
  }

  static bool all_clear(void);

 private:
  Allocator() = default;
  ~Allocator() = default;
  static Allocator& singleton() {
    static Allocator Allocator;
    return Allocator;
  };
  static void* allocate(size_t size);
  static void deallocate(void* ptr, size_t size);

  static size_t allocate_memory_size_;
  static size_t deallocate_memory_size_;

  struct free_deletor {
    void operator()(void* ptr) { std::free(ptr); }
  };

  /* cache_ saves all of the pointers that have been deallocated.
     So we can reuse it instead of malloc again */
  std::multimap<size_t, std::unique_ptr<void, free_deletor>> cache_;
};

}  // namespace tinytorch
