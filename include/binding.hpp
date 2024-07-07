/**
 * Copyright (c) 2022-2024 yewentao256
 * Licensed under the MIT License.
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "array.hpp"

namespace py = pybind11;

namespace pybind11 {
namespace detail {
template <>
struct type_caster<microtorch::IntArrayRef> {
  PYBIND11_TYPE_CASTER(microtorch::IntArrayRef, const_name("IntArrayRef"));

  // Python -> C++
  bool load(handle src, bool) {
    if (!py::isinstance<py::sequence>(src) || py::isinstance<py::str>(src)) {
      return false;
    }
    value = microtorch::IntArrayRef(src.cast<std::vector<int64_t>>());
    return true;
  }

  // C++ -> Python
  static handle cast(const microtorch::IntArrayRef &src,
                     return_value_policy policy, handle parent) {
    return py::cast(src.vec(), policy, parent).release();
  }
};
}  // namespace detail
}  // namespace pybind11

void export_tensor_class(py::module &m);
void export_tensor_function(py::module &m);
void export_optimizer_class(py::module &m);
void export_test_function(py::module &m);
void export_cuda_function(py::module &m);
void export_graph_function(py::module &m);
