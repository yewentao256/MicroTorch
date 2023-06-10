#include <pybind11/pybind11.h>

#include "../src/unit_test.hpp"

namespace py = pybind11;
using namespace tinytorch;

void export_test_function(py::module &m) {
  m.def("unit_test", &tinytorch::unit_test, "c++ unit test");
}
