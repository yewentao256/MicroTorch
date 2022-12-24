#include <pybind11/pybind11.h>

#include "../src/graph.hpp"
#include "../src/backward.hpp"

namespace py = pybind11;
using namespace tinytorch;

void export_graph_function(py::module &m) {
  m.def("make_parameter", &tinytorch::makeParameter, "initialize the params")
  .def("backward", &tinytorch::backward, "backward function");
}
