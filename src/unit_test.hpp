#pragma once
#include "allocator.hpp"
#include "exception.hpp"
#include <iostream>
#include "storage.hpp"
#include <chrono>

using namespace std::chrono;
namespace tinytorch {

const char* test_device() {
  Device device = Device("cpu");
  Device device_cuda = Device(DeviceType::CUDA);
  
  TORCH_CHECK(!(device.is_cuda()), "check 1");
  TORCH_CHECK(device.is_cpu(), "check 1");
  TORCH_CHECK(device.str() == "cpu", "check1");

  TORCH_CHECK(device_cuda != device, "check2");
  TORCH_CHECK(device_cuda == Device("cuda"), "check2");

  return "passed!";
}

struct Object {
  static int constructor_call_counter;
  static int deconstructor_call_counter;

  char x_;
  char y_;
  Object() : x_('0'), y_('0') { constructor_call_counter++; }
  Object(char x, char y) : x_(x), y_(y) { constructor_call_counter++; }
  ~Object() { deconstructor_call_counter++; }
  friend std::ostream& operator<<(std::ostream& os, const Object& obj) {
    os << "Object(" << obj.x_ << ", " << obj.y_ << ")";
    return os;
  }
};
int Object::constructor_call_counter = 0;
int Object::deconstructor_call_counter = 0;

const char* test_cpu_allocator() {
  auto allocator = g_allocator_manager.get_allocator(Device("cpu"));

  void* ptr;
  {
    // No constructor call here. Firstly Allocator will std::malloc memory
    auto uptr = allocator->unique_allocate<Object>(sizeof(Object));
    TORCH_CHECK(Object::constructor_call_counter == 0, "check 1");
    ptr = uptr.get();
  }
  TORCH_CHECK(Object::deconstructor_call_counter == 0, "check 1");

  {
    // The strategy of Allocator. Here we reuse the pointer above
    auto sptr = allocator->shared_allocate<Object>(sizeof(Object));
    TORCH_CHECK(ptr == static_cast<void*>(sptr.get()), "check 2");
  }

  {
    // Construct the object really
    auto uptr = allocator->unique_construct<Object>();
    TORCH_CHECK(Object::constructor_call_counter == 1, "check 3");
    TORCH_CHECK(ptr == static_cast<void*>(uptr.get()), "check 3");
  }
  TORCH_CHECK(Object::deconstructor_call_counter == 1, "check 3");

  {
    auto sptr = allocator->shared_construct<Object>('6', '7');
    TORCH_CHECK(Object::constructor_call_counter == 2, "check 4");
    TORCH_CHECK(sptr->x_ == '6' && sptr->y_ == '7', "check 4");
    TORCH_CHECK(ptr == static_cast<void*>(sptr.get()), "check 4");
  }
  TORCH_CHECK(Object::deconstructor_call_counter == 2, "check 4");

  TORCH_CHECK(allocator->all_clear(), "check memory all clear fail");
  return "passed!";
}


const char* test_cuda_allocator() {
  auto allocator = g_allocator_manager.get_allocator(Device("cuda"));

  void* ptr;
  {
    // No constructor call here.
    auto uptr = allocator->unique_allocate<Object>(sizeof(Object));
    ptr = uptr.get();
  }

  {
    // The strategy of Allocator. Here we reuse the pointer above
    auto sptr = allocator->shared_allocate<Object>(sizeof(Object));
    TORCH_CHECK(ptr == static_cast<void*>(sptr.get()), "check 2");
  }

  TORCH_CHECK(allocator->all_clear(), "check memory all clear fail");
  return "passed!";
}

int unit_test() {
  steady_clock::time_point start_tp = steady_clock::now();
  std::cout << "test Device...  \033[32m" << test_device() << "\33[0m" << std::endl;

  std::cout << "test Allocator[cpu]...  \033[32m" << test_cpu_allocator() << "\33[0m" << std::endl;
#ifdef USE_CUDA
  std::cout << "test Allocator[cuda]...  \033[32m" << test_cuda_allocator() << "\33[0m" << std::endl;
#endif

  steady_clock::time_point end_tp = steady_clock::now();
  duration<double> time_span =
      duration_cast<duration<double>>(end_tp - start_tp);
  std::cout << "\033[32mTest success. Test took " << time_span.count();
  std::cout << " seconds.\033[0m" << std::endl;
  return 0;
}
} // namespace tinytorch
