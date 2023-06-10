#pragma once
#include "allocator.hpp"
#include "exception.hpp"
#include <iostream>
#include "storage.hpp"
#include <chrono>

using namespace std::chrono;
namespace tinytorch {

struct Object {
  static int ctr_call_counter;
  static int dectr_call_counter;

  char x_;
  char y_;
  Object() : x_('0'), y_('0') { ++ctr_call_counter; }
  Object(char x, char y) : x_(x), y_(y) { ++ctr_call_counter; }
  ~Object() { ++dectr_call_counter; }
  friend std::ostream& operator<<(std::ostream& os, const Object& obj) {
    os << "Object(" << obj.x_ << ", " << obj.y_ << ")";
    return os;
  }
};
int Object::ctr_call_counter = 0;
int Object::dectr_call_counter = 0;

const char * test_Alloc() {
  void* ptr;
  {
    // No constructor call here. Firstly allocator will std::malloc memory
    auto uptr = Allocator::unique_allocate<Object>(sizeof(Object));
    TORCH_CHECK(Object::ctr_call_counter == 0, "check 1");
    ptr = uptr.get();
  }
  TORCH_CHECK(Object::dectr_call_counter == 0, "check 1");

  {
    // The strategy of allocator. Here we reuse the pointer above
    auto sptr = Allocator::shared_allocate<Object>(sizeof(Object));
    TORCH_CHECK(ptr == static_cast<void*>(sptr.get()), "check 2");
  }

  {
    // Construct the object really
    auto uptr = Allocator::unique_construct<Object>();
    TORCH_CHECK(Object::ctr_call_counter == 1, "check 3");
    TORCH_CHECK(ptr == static_cast<void*>(uptr.get()), "check 3");
  }
  TORCH_CHECK(Object::dectr_call_counter == 1, "check 3");

  {
    auto sptr = Allocator::shared_construct<Object>('6', '7');
    TORCH_CHECK(Object::ctr_call_counter == 2, "check 4");
    TORCH_CHECK(sptr->x_ == '6' && sptr->y_ == '7', "check 4");
    TORCH_CHECK(ptr == static_cast<void*>(sptr.get()), "check 4");
  }
  TORCH_CHECK(Object::dectr_call_counter == 2, "check 4");
  return "passed!";
}

int unit_test() {
  Device device("cuda");
  std::cout << "hello!" << device << std::endl;

  steady_clock::time_point start_tp = steady_clock::now();
  std::cout << "test allocator...  \033[32m" << test_Alloc() << "\33[0m" << std::endl;

  std::cout << "check all memory is deallocated...  \033[32m";
  TORCH_CHECK(Allocator::all_clear(), "check memory all clear");
  std::cout << "passed!\33[0m" << std::endl;

  steady_clock::time_point end_tp = steady_clock::now();
  duration<double> time_span =
      duration_cast<duration<double>>(end_tp - start_tp);
  std::cout << "\033[32mTest success. Test took " << time_span.count();
  std::cout << " seconds.\033[0m" << std::endl;
  return 0;
}
} // namespace tinytorch
