#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "optimizer.hpp"

namespace py = pybind11;
using namespace microtorch;

void export_optimizer_class(py::module &m) {
  py::class_<SGDOptimizer>(m, "SGDOptimizer")
      // init function
      .def(py::init<std::vector<Tensor>, float, float, float>(),
           py::arg("params"), py::arg("lr"), py::arg_v("momentum", 0.0f),
           py::arg_v("dampening", 0.0f))

      // python specs
      .def("__repr__", [](Tensor &t) { return "<microtorch.Optimizer object>"; })

      // functions
      .def("zero_grad", &SGDOptimizer::zeroGrad)
      .def("step", &SGDOptimizer::step);
}
