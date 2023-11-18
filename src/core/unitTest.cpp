/**
 * Copyright (c) 2022-2023 yewentao
 * Licensed under the MIT License.
 */
#include "unitTest.hpp"

#include <chrono>
#include <iostream>
#include <type_traits>

#include "allocator.hpp"
#include "exception.hpp"
#include "funcRef.hpp"
#include "funcTraits.hpp"
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

class TestClass {
 public:
  short memberFunction(int, double, float) { return 0; }
  float constMemberFunction(int, double) const { return 1.0; }
};
void normalFunction(int) {}
auto lambdaFunction = []() -> void {};

const char* test_func_traits() {
  // test normal function
  using traits = FuncTraits<decltype(normalFunction)>;
  TORCH_CHECK(traits::num_args == 1, "Function num_args test failed.");
  // arg<i>::type depends on template <size_t i>, so we should add `typename`
  TORCH_CHECK((std::is_same_v<typename traits::arg<0>::type, int>),
              "check argument type failed.");

  // test class member
  using traits2 = FuncTraits<decltype(&TestClass::memberFunction)>;
  TORCH_CHECK(traits2::num_args == 3, "Member function num_args test failed");
  TORCH_CHECK((std::is_integral_v<traits2::result_type>),
              "Member function return type test failed");

  // test const class member
  using traits3 = FuncTraits<decltype(&TestClass::constMemberFunction)>;
  TORCH_CHECK(traits3::num_args == 2,
              "Const member function num_args test failed");
  TORCH_CHECK((std::is_same_v<typename traits3::arg<1>::type, double>),
              "Argument type failed.");

  // test lambda (const)
  using traits4 = FuncTraits<decltype(lambdaFunction)>;
  TORCH_CHECK(traits4::num_args == 0, "Lambda function num_args test failed");

  // test lambda (mutable)
  auto mutableLambda = [&]() mutable {};
  using traits5 = FuncTraits<decltype(mutableLambda)>;
  TORCH_CHECK((std::is_void_v<traits5::result_type>),
              "Lambda function return type test failed");

  // test ref
  using traits_ref = FuncTraits<decltype((void (&)(int))normalFunction)>;
  TORCH_CHECK(traits_ref::num_args == 1,
              "Reference function num_args test failed");

  // test pointer
  using traits_ptr = FuncTraits<decltype(&normalFunction)>;
  TORCH_CHECK((std::is_same_v<traits_ptr::arg<0>::type, int>),
              "Pointer function argument type test failed");
  return "passed!";
}

int add(int a, int b) { return a + b; }

const char* test_func_ref() {
  microtorch::FuncRef<int(int, int)> funcRef = add;
  TORCH_CHECK(funcRef(2, 3) == 5, "normal function");

  auto square = [](int x) { return x * x; };
  microtorch::FuncRef<int(int)> lambdaRef = square;
  TORCH_CHECK(lambdaRef(4) == 16, "lambda function");

  TestClass testObj;
  // use std::bind to make a `Callable`
  auto memberFunc =
      std::bind(&TestClass::memberFunction, &testObj, std::placeholders::_1,
                std::placeholders::_2, std::placeholders::_3);
  microtorch::FuncRef<short(int, double, float)> memberRef = memberFunc;
  TORCH_CHECK(memberRef(1, 2.0, 3.0f) == 0, "member function");

  auto constMemberFunc =
      std::bind(&TestClass::constMemberFunction, &testObj,
                std::placeholders::_1, std::placeholders::_2);
  microtorch::FuncRef<float(int, double)> constMemberRef = constMemberFunc;
  TORCH_CHECK(constMemberRef(1, 2.0) == 1.0f, "const member function");

  microtorch::FuncRef<int(int, int)> nullRef = nullptr;
  TORCH_CHECK(!nullRef, "nullptr function");

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
  std::cout << "test FuncTraits ...  \033[32m" << test_func_traits() << "\33[0m"
            << std::endl;
  std::cout << "test FuncRef ...  \033[32m" << test_func_ref() << "\33[0m"
            << std::endl;
  steady_clock::time_point end_tp = steady_clock::now();
  duration<double> time_span =
      duration_cast<duration<double>>(end_tp - start_tp);
  std::cout << "\033[32m All of the cpp unit tests success. Test took "
            << time_span.count() << " seconds.\033[0m" << std::endl;
  return 1;
}
}  // namespace microtorch
