#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../src/optimizer.hpp"

namespace py = pybind11;
using namespace tinytorch;

void export_optimizer_class(py::module &m) {
  py::class_<SGDOptimizer>(m, "SGDOptimizer")
      // init function
      .def(py::init<std::vector<Tensor>, float>(), py::arg("params"), py::arg("lr"))

      // python specs
      .def("__repr__",
           [](Tensor &t) {
             return "<tinytorch.Optimizer object>";
           })
      
      // functions
      .def("zero_grad", &SGDOptimizer::zeroGrad)
      .def("step", &SGDOptimizer::step);
}
