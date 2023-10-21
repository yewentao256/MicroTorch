/**
 * Copyright (c) 2022-2023 yewentao
 * Licensed under the MIT License.
 */
#include "binding.hpp"
#include "unitTest.hpp"

#ifdef USE_CUDA
#define CUDA_AVAILABLE true
#else
#define CUDA_AVAILABLE false
#endif

using namespace microtorch;

void export_test_function(py::module &m) {
  m.def("unit_test", &microtorch::unit_test, "C++ unit test funcs");
}
