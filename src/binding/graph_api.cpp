/**
 * Copyright (c) 2022-2024 yewentao256
 * Licensed under the MIT License.
 */
#include "binding.hpp"
#include "graph.hpp"

using namespace microtorch;

void export_graph_function(py::module& m) {
  m.def("set_grad_mode", &GradModeController::set_enabled, "Set grad mode");
  m.def("is_grad_mode_enabled", &GradModeController::is_enabled,
        "Check if grad mode is enabled");

  py::class_<AutoGradGuard>(m, "AutoGradGuard")
      .def(py::init<bool>(), py::arg("enabled") = true)
      .def("__enter__",
           [](AutoGradGuard& self) -> AutoGradGuard& { return self; })
      .def("__exit__", [](AutoGradGuard&, const py::object&, const py::object&,
                          const py::object&) {});
}
