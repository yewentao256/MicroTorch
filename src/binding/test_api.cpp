#include <pybind11/pybind11.h>

#include "unitTest.hpp"

#ifdef USE_CUDA
#define CUDA_AVAILABLE true
#else
#define CUDA_AVAILABLE false
#endif

namespace py = pybind11;
using namespace microtorch;

void export_test_function(py::module &m) {
  m.def("unit_test", &microtorch::unit_test, "C++ unit test funcs")
      .def(
          "is_cuda_available", []() { return CUDA_AVAILABLE; },
          "Check if CUDA is available");
}
