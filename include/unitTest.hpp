/**
 * Copyright (c) 2022-2023 yewentao
 * Licensed under the MIT License.
 */
#pragma once
#include <chrono>
#include <iostream>

#include "allocator.hpp"
#include "exception.hpp"
#include "storage.hpp"

using namespace std::chrono;
namespace microtorch {

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
  Object(char x, char y) : x_(x), y_(y) { constructor_call_counter++; }
  ~Object() { deconstructor_call_counter++; }
  friend std::ostream& operator<<(std::ostream& os, const Object& obj) {
    os << "Object(" << obj.x_ << ", " << obj.y_ << ")";
    return os;
  }
};
int Object::constructor_call_counter = 0;
int Object::deconstructor_call_counter = 0;

const char* test_allocator(const Device& device) {
  g_allocator_manager.reset_allocators();
  auto allocator = g_allocator_manager.get_allocator(device);

  void* ptr;
  {
    // No constructor is called. Firstly Allocator will malloc memory
    auto uptr = allocator->shared_allocate<Object>(sizeof(Object));
    TORCH_CHECK(Object::constructor_call_counter == 0, "check 1");
    ptr = uptr.get();
  }
  // Accordingly, no deconstructor is called
  TORCH_CHECK(Object::deconstructor_call_counter == 0, "check 1");

  {
    // The size-match strategy of Allocator. Here we reuse the pointer before
    auto sptr = allocator->unique_allocate<Object>(sizeof(Object));
    TORCH_CHECK(ptr == static_cast<void*>(sptr.get()), "check 2");
  }

  TORCH_CHECK(allocator->check_all_clear(), "check memory all clear fail");
  return "passed!";
}

int unit_test() {
  steady_clock::time_point start_tp = steady_clock::now();
  std::cout << "test Device...  \033[32m" << test_device() << "\33[0m"
            << std::endl;

  std::cout << "test Allocator[cpu]...  \033[32m"
            << test_allocator(Device("cpu")) << "\33[0m" << std::endl;
#ifdef USE_CUDA
  std::cout << "test Allocator[cuda]...  \033[32m"
            << test_allocator(Device("cuda")) << "\33[0m" << std::endl;
#endif

  steady_clock::time_point end_tp = steady_clock::now();
  duration<double> time_span =
      duration_cast<duration<double>>(end_tp - start_tp);
  std::cout << "\033[32m All of the cpp unit tests success. Test took "
            << time_span.count() << " seconds.\033[0m" << std::endl;
  return 1;
}
}  // namespace microtorch
