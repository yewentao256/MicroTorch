#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../src/ops.hpp"
#include "../src/tensor2.hpp"
namespace py = pybind11;
using namespace tinytorch;

void export_tensor2_class(py::module &m) {
  py::class_<Tensor2>(m, "Tensor2", py::buffer_protocol())
      // init function
      .def(py::init([](py::buffer b) {
        /* Request a buffer descriptor from Python */
        py::buffer_info info = b.request();

        /* Some basic validation checks ... */
        if (info.format != py::format_descriptor<data_t>::format())
          throw std::runtime_error(
              "Incompatible format: expected a float array!");

        return Tensor2(static_cast<data_t *>(info.ptr), (index_t) info.ndim, (index_t *)info.shape.data());
      }))

      .def(
          "__getitem__",
          [](Tensor2 &t, std::vector<index_t> idxs) { return t[idxs]; },
          py::is_operator());
}
